'use strict';

// Edge-IDS v2.0 Dashboard — 自动刷新 + 攻击记录 + 实时曲线图 + 攻击类型饼图

// ========== 攻击类型配色体系 ==========
const ATTACK_TYPE_COLORS = {
    'DoS':              '#ef4444',  // 严重-红
    'Exploits':         '#f97316',  // 严重-橙红
    'Backdoor':         '#b91c1c',  // 严重-暗红
    'Shellcode':        '#a855f7',  // 严重-紫
    'Worms':            '#7c3aed',  // 严重-深紫
    'Fuzzers':          '#f59e0b',  // 高危-橙
    'Generic':          '#eab308',  // 高危-黄
    'Reconnaissance':   '#3b82f6',  // 中危-蓝
    'Analysis':         '#06b6d4',  // 中危-青
};

/** 攻击类型 → 危险等级映射 */
const ATTACK_DANGER_MAP = {
    'DoS':              '严重',
    'Exploits':         '严重',
    'Backdoor':         '严重',
    'Shellcode':        '严重',
    'Worms':            '严重',
    'Fuzzers':          '高危',
    'Generic':          '高危',
    'Reconnaissance':   '中危',
    'Analysis':         '中危',
};

/** 饼图中展示的所有攻击类型（按固定顺序） */
const PIE_CHART_LABELS = Object.keys(ATTACK_TYPE_COLORS);
const PIE_CHART_COLORS = Object.values(ATTACK_TYPE_COLORS);

// ========== 状态变量 ==========
let refreshTimer = null;
let attackTimer = null;
let chartTimer = null;
let fetchErrors = 0;
let lastAttackCount = 0;
let lastSeenTime = '';
let timelineChart = null;
let attackTypeChart = null;
let isReconnecting = false;

const NORMAL_INTERVAL = 2000;       // 正常轮询间隔 2 秒
const RECONNECT_INTERVAL = 10000;   // 断线重连间隔 10 秒
const MAX_ERRORS = 5;               // 触发重连的错误阈值

// ========== 初始化 ==========

function initDashboard() {
    fetchStatus();
    fetchAttacks();

    // 图表初始化放在 setTimeout 确保 DOM 就绪，失败不影响其他功能
    setTimeout(function () {
        try {
            initTimelineChart();
            initPieChart();
            chartTimer = setInterval(fetchTimeSeries, 1000);
            fetchTimeSeries();
        } catch (e) {
            console.error('图表初始化失败:', e);
        }
    }, 100);

    refreshTimer = setInterval(fetchStatus, NORMAL_INTERVAL);
    attackTimer = setInterval(fetchAttacks, NORMAL_INTERVAL);
}

// ========== 定时器管理 ==========

/** 恢复正常轮询频率 */
function resetTimersToNormal() {
    if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
    if (attackTimer) { clearInterval(attackTimer); attackTimer = null; }
    refreshTimer = setInterval(fetchStatus, NORMAL_INTERVAL);
    attackTimer = setInterval(fetchAttacks, NORMAL_INTERVAL);
    isReconnecting = false;
}

/** 切换到降频重连模式 */
function slowDownTimers() {
    if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null; }
    if (attackTimer) { clearInterval(attackTimer); attackTimer = null; }
    refreshTimer = setInterval(fetchStatus, RECONNECT_INTERVAL);
    attackTimer = setInterval(fetchAttacks, RECONNECT_INTERVAL);
    isReconnecting = true;
    const badge = document.getElementById('status-badge');
    if (badge) {
        badge.textContent = '重连中...';
        badge.className = 'badge reconnecting';
    }
}

// ========== 系统状态 ==========

async function fetchStatus() {
    try {
        const resp = await fetch('/api/stats');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        fetchErrors = 0;

        // 重连成功后恢复
        if (isReconnecting) {
            resetTimersToNormal();
        }

        const s = data.status;
        const sys = data.system;
        const res = data.resources;

        setText('packets-captured', s.packets_captured || 0);
        setText('packets-dropped', s.packets_dropped || 0);
        setText('flows-active', s.flows_active || 0);
        setText('flows-analyzed', s.flows_analyzed || 0);
        setText('attacks-detected', s.attacks_detected || 0);
        setText('attacks-total', s.attacks_total || 0);
        setText('avg-latency', (s.avg_latency_ms || 0).toFixed(2) + ' ms');

        const badge = document.getElementById('status-badge');
        if (badge && !isReconnecting) {
            if (s.is_running) {
                badge.textContent = '运行中';
                badge.className = 'badge running';
            } else {
                badge.textContent = '已停止';
                badge.className = 'badge stopped';
            }
        }

        if (sys) {
            setText('sys-platform', sys.platform || '-');
        }
        if (res) {
            setText('sys-memory', (res.memory_percent || 0).toFixed(1) + '%');
            setText('sys-cpu', (res.cpu_percent || 0).toFixed(1) + '%');
        }

        if (s.start_time) {
            setText('uptime', s.start_time);
        }
    } catch (e) {
        fetchErrors++;
        if (fetchErrors >= MAX_ERRORS && !isReconnecting) {
            slowDownTimers();
        }
    }
}

// ========== 攻击记录 ==========

async function fetchAttacks() {
    try {
        const resp = await fetch('/api/attacks?limit=100');
        if (!resp.ok) return;
        const data = await resp.json();

        setText('log-count', data.total || 0);

        if (data.total > lastAttackCount && lastAttackCount > 0) {
            flashAlert(data.total - lastAttackCount);
        }
        lastAttackCount = data.total;

        if (data.danger_counts) {
            setText('d-severe', data.danger_counts['严重'] || 0);
            setText('d-high', data.danger_counts['高危'] || 0);
            setText('d-medium', data.danger_counts['中危'] || 0);
            setText('d-low', data.danger_counts['低危'] || 0);
        }

        if (data.type_counts && Object.keys(data.type_counts).length > 0) {
            document.getElementById('type-stats-section').style.display = 'block';
            updatePieChart(data.type_counts);
        }

        if (data.attacks && data.attacks.length > 0) {
            renderAttackTable(data.attacks);
        }
    } catch (e) { /* 静默失败，不影响状态轮询 */ }
}

function flashAlert(count) {
    const el = document.getElementById('alert-flash');
    if (!el) return;
    el.textContent = '⚠ 检测到 ' + count + ' 次新攻击!';
    el.classList.remove('hidden');
    setTimeout(function () { el.classList.add('hidden'); }, 3000);
}

/** 使用 textContent 渲染表格，防止 XSS */
function renderAttackTable(attacks) {
    const tbody = document.querySelector('#attack-table tbody');
    if (!tbody) return;

    let hasNew = false;
    if (attacks.length > 0 && attacks[0].time !== lastSeenTime) {
        hasNew = true;
        lastSeenTime = attacks[0].time;
    }

    // 清空现有行
    tbody.innerHTML = '';

    const limit = Math.min(attacks.length, 100);
    for (let i = 0; i < limit; i++) {
        const a = attacks[i];
        const tr = document.createElement('tr');
        if (hasNew && i === 0) {
            tr.className = 'new-row';
        }

        // 时间
        const tdTime = document.createElement('td');
        tdTime.textContent = a.time || '-';
        tr.appendChild(tdTime);

        // 来源
        const tdSrc = document.createElement('td');
        tdSrc.textContent = a.src || '-';
        tr.appendChild(tdSrc);

        // 目标
        const tdDst = document.createElement('td');
        tdDst.textContent = a.dst || '-';
        tr.appendChild(tdDst);

        // 协议
        const tdProto = document.createElement('td');
        tdProto.textContent = a.protocol || '-';
        tr.appendChild(tdProto);

        // 攻击类型（带颜色 tag）
        const tdType = document.createElement('td');
        tdType.appendChild(createAttackTypeTag(a.type || '未知'));
        tr.appendChild(tdType);

        // 置信度
        const tdConf = document.createElement('td');
        const confClass = a.confidence >= 0.95 ? 'conf-high'
            : (a.confidence >= 0.88 ? 'conf-mid' : 'conf-low');
        const confSpan = document.createElement('span');
        confSpan.className = confClass;
        confSpan.textContent = (a.confidence * 100).toFixed(1) + '%';
        tdConf.appendChild(confSpan);
        tr.appendChild(tdConf);

        // 危险等级
        const tdDanger = document.createElement('td');
        tdDanger.appendChild(createDangerTag(a.danger || '低危'));
        tr.appendChild(tdDanger);

        tbody.appendChild(tr);
    }
}

// ========== 饼图（攻击类型 Doughnut） ==========

function initPieChart() {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js 未加载，饼图不可用');
        return;
    }
    const ctx = document.getElementById('attackTypeChart');
    if (!ctx) return;

    attackTypeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: PIE_CHART_LABELS,
            datasets: [{
                data: PIE_CHART_LABELS.map(function () { return 0; }),
                backgroundColor: PIE_CHART_COLORS,
                borderColor: '#1e293b',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 11 },
                        padding: 12,
                        usePointStyle: true,
                    },
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#e2e8f0',
                    bodyColor: '#e2e8f0',
                    borderColor: '#334155',
                    borderWidth: 1,
                },
            },
        },
    });
}

/** 根据 type_counts 更新饼图数据 */
function updatePieChart(typeCounts) {
    if (!attackTypeChart) return;
    const data = PIE_CHART_LABELS.map(function (label) {
        return typeCounts[label] || 0;
    });
    attackTypeChart.data.datasets[0].data = data;
    attackTypeChart.update('none');
}

// ========== 实时曲线图 ==========

function initTimelineChart() {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js 未加载，曲线图不可用');
        return;
    }
    const ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    const gridColor = 'rgba(148, 163, 184, 0.12)';
    const textColor = '#94a3b8';

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '流量 (pps)',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.08)',
                    borderWidth: 1.8,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y',
                    hidden: false,
                },
                {
                    label: '延迟 (ms)',
                    data: [],
                    borderColor: '#fbbf24',
                    borderWidth: 1.8,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y',
                    hidden: false,
                },
                {
                    label: '攻击频率 (次/s)',
                    data: [],
                    borderColor: '#f87171',
                    backgroundColor: 'rgba(248, 113, 113, 0.06)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y',
                    hidden: false,
                },
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: '#a78bfa',
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y1',
                    hidden: true,
                },
                {
                    label: '内存 %',
                    data: [],
                    borderColor: '#34d399',
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y1',
                    hidden: true,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 200 },
            interaction: {
                intersect: false,
                mode: 'nearest',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#38bdf8',
                    bodyColor: '#e2e8f0',
                    borderColor: '#334155',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    ticks: {
                        color: textColor,
                        maxTicksLimit: 10,
                        maxRotation: 0,
                        font: { size: 10 },
                    },
                    grid: { color: gridColor },
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    ticks: {
                        color: textColor,
                        font: { size: 10 },
                    },
                    grid: { color: gridColor },
                    title: {
                        display: true,
                        text: 'pps / ms / 次',
                        color: textColor,
                        font: { size: 10 },
                    },
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: textColor,
                        font: { size: 10 },
                        callback: function (v) { return v + '%'; },
                    },
                    grid: { drawOnChartArea: false },
                    title: {
                        display: true,
                        text: '百分比 %',
                        color: textColor,
                        font: { size: 10 },
                    },
                },
            },
        },
    });
}

async function fetchTimeSeries() {
    if (!timelineChart) return;
    try {
        const resp = await fetch('/api/timeseries?limit=120');
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data || !data.length) return;

        const labels = [];
        const trafficData = [];
        const latencyData = [];
        const attacksData = [];
        const cpuData = [];
        const memData = [];

        for (let i = 0; i < data.length; i++) {
            const d = data[i];
            labels.push(d.time);

            // 流量：后端已计算 pps
            trafficData.push(d.packets_sec || 0);

            // 延迟
            latencyData.push(d.avg_latency_ms || 0);

            // 攻击频率：用 attacks_total 差值
            let atkDelta = 0;
            if (i > 0) {
                atkDelta = Math.max(0, d.attacks_total - data[i - 1].attacks_total);
            }
            attacksData.push(atkDelta);

            // 系统负载
            cpuData.push(d.cpu_percent || 0);
            memData.push(d.memory_percent || 0);
        }

        timelineChart.data.labels = labels;
        timelineChart.data.datasets[0].data = trafficData;
        timelineChart.data.datasets[1].data = latencyData;
        timelineChart.data.datasets[2].data = attacksData;
        timelineChart.data.datasets[3].data = cpuData;
        timelineChart.data.datasets[4].data = memData;
        timelineChart.update('none');
    } catch (e) { /* 静默失败 */ }
}

function toggleChart(key, checkbox) {
    if (!timelineChart) return;
    if (key === 'system') {
        // 系统负载: 切换 CPU(3) + 内存(4) 两条线
        timelineChart.setDatasetVisibility(3, checkbox.checked);
        timelineChart.setDatasetVisibility(4, checkbox.checked);
    } else if (key === 'traffic') {
        timelineChart.setDatasetVisibility(0, checkbox.checked);
    } else if (key === 'latency') {
        timelineChart.setDatasetVisibility(1, checkbox.checked);
    } else if (key === 'attacks') {
        timelineChart.setDatasetVisibility(2, checkbox.checked);
    }
    timelineChart.update('none');
}

// ========== DOM 辅助函数 ==========

/** 创建危险等级 tag（使用 textContent 防 XSS） */
function createDangerTag(level) {
    const span = document.createElement('span');
    span.className = 'tag';
    if (level === '严重') {
        span.classList.add('severe');
    } else if (level === '高危') {
        span.classList.add('high');
    } else if (level === '中危') {
        span.classList.add('medium');
    } else {
        span.classList.add('low');
    }
    span.textContent = level;
    return span;
}

/** 创建攻击类型彩色 tag */
function createAttackTypeTag(typeName) {
    const span = document.createElement('span');
    span.className = 'attack-type-tag';
    span.style.backgroundColor = ATTACK_TYPE_COLORS[typeName] || '#64748b';
    span.textContent = typeName;
    return span;
}

/** 设置元素文本（仅在值变化时更新，减少 DOM 操作） */
function setText(id, text) {
    const el = document.getElementById(id);
    const strVal = String(text);
    if (el && el.textContent !== strVal) {
        el.textContent = strVal;
    }
}

// ========== 控制按钮 ==========

async function control(action) {
    try {
        const resp = await fetch('/api/control/' + action, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            setTimeout(fetchStatus, 300);
            setTimeout(fetchAttacks, 500);
        } else {
            alert('操作失败: ' + (data.error || '未知错误'));
        }
    } catch (e) {
        alert('请求失败: ' + e.message);
    }
}
