// Edge-IDS v2.0 Dashboard — 自动刷新 + 攻击记录 + 实时曲线图
let refreshTimer = null;
let attackTimer = null;
let chartTimer = null;
let fetchErrors = 0;
let lastAttackCount = 0;
let lastSeenTime = '';
let timelineChart = null;

function initDashboard() {
    fetchStatus();
    fetchAttacks();
    // 图表初始化放在 setTimeout 确保 DOM 就绪，且失败不影响其他功能
    setTimeout(function () {
        try {
            initChart();
            chartTimer = setInterval(fetchTimeSeries, 1000);
            fetchTimeSeries();
        } catch (e) {
            console.error('图表初始化失败:', e);
        }
    }, 100);
    refreshTimer = setInterval(fetchStatus, 2000);
    attackTimer = setInterval(fetchAttacks, 2000);
}

// ========== 系统状态 ==========

async function fetchStatus() {
    try {
        var resp = await fetch('/api/stats');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var data = await resp.json();
        fetchErrors = 0;

        var s = data.status;
        var sys = data.system;
        var res = data.resources;

        setText('packets-captured', s.packets_captured || 0);
        setText('packets-dropped', s.packets_dropped || 0);
        setText('flows-active', s.flows_active || 0);
        setText('flows-analyzed', s.flows_analyzed || 0);
        setText('attacks-detected', s.attacks_detected || 0);
        setText('attacks-total', s.attacks_total || 0);
        setText('avg-latency', (s.avg_latency_ms || 0).toFixed(2) + ' ms');

        var badge = document.getElementById('status-badge');
        if (badge) {
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
        if (fetchErrors > 5) {
            clearInterval(refreshTimer);
            refreshTimer = null;
            var badge = document.getElementById('status-badge');
            if (badge) {
                badge.textContent = '连接断开';
                badge.className = 'badge stopped';
            }
        }
    }
}

// ========== 攻击记录 ==========

async function fetchAttacks() {
    try {
        var resp = await fetch('/api/attacks?limit=100');
        if (!resp.ok) return;
        var data = await resp.json();

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
            renderTypeStats(data.type_counts);
        }

        if (data.attacks && data.attacks.length > 0) {
            renderAttackTable(data.attacks);
        }
    } catch (e) { /* silent fail */ }
}

function flashAlert(count) {
    var el = document.getElementById('alert-flash');
    if (!el) return;
    el.textContent = '⚠ 检测到 ' + count + ' 次新攻击!';
    el.classList.remove('hidden');
    setTimeout(function () { el.classList.add('hidden'); }, 3000);
}

function renderTypeStats(counts) {
    var container = document.getElementById('type-stats');
    var html = '';
    var sorted = Object.entries(counts).sort(function (a, b) { return b[1] - a[1]; });
    for (var i = 0; i < sorted.length; i++) {
        html += '<div class="type-chip"><span class="t-name">' + sorted[i][0] +
                '</span><span class="t-count">' + sorted[i][1] + '</span></div>';
    }
    container.innerHTML = html;
}

function renderAttackTable(attacks) {
    var tbody = document.querySelector('#attack-table tbody');
    var hasNew = false;
    if (attacks.length > 0 && attacks[0].time !== lastSeenTime) {
        hasNew = true;
        lastSeenTime = attacks[0].time;
    }

    var html = '';
    for (var i = 0; i < Math.min(attacks.length, 100); i++) {
        var a = attacks[i];
        var rowClass = (hasNew && i === 0) ? 'new-row' : '';
        var confClass = a.confidence >= 0.95 ? 'conf-high' : (a.confidence >= 0.88 ? 'conf-mid' : 'conf-low');
        var pct = (a.confidence * 100).toFixed(1);
        var dangerTag = dangerTagHtml(a.danger);
        html += '<tr class="' + rowClass + '">' +
            '<td>' + (a.time || '-') + '</td>' +
            '<td>' + (a.src || '-') + '</td>' +
            '<td>' + (a.dst || '-') + '</td>' +
            '<td>' + (a.protocol || '-') + '</td>' +
            '<td>' + (a.type || '未知') + '</td>' +
            '<td><span class="' + confClass + '">' + pct + '%</span></td>' +
            '<td>' + dangerTag + '</td>' +
            '</tr>';
    }
    tbody.innerHTML = html;
}

function dangerTagHtml(level) {
    var cls = '';
    if (level === '严重') cls = 'severe';
    else if (level === '高危') cls = 'high';
    else if (level === '中危') cls = 'medium';
    else cls = 'low';
    return '<span class="tag ' + cls + '">' + level + '</span>';
}

// ========== 实时曲线图 ==========

function initChart() {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js 未加载，图表不可用');
        return;
    }
    var ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    var gridColor = 'rgba(148, 163, 184, 0.12)';
    var textColor = '#94a3b8';

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
        var resp = await fetch('/api/timeseries?limit=120');
        if (!resp.ok) return;
        var data = await resp.json();
        if (!data || !data.length) return;

        var labels = [];
        var trafficData = [];
        var latencyData = [];
        var attacksData = [];
        var cpuData = [];
        var memData = [];

        var prevAttacks = data.length > 0 ? data[0].attacks_total : 0;

        for (var i = 0; i < data.length; i++) {
            var d = data[i];
            labels.push(d.time);

            // 流量：后端已计算 pps
            trafficData.push(d.packets_sec || 0);

            // 延迟
            latencyData.push(d.avg_latency_ms || 0);

            // 攻击频率：用 attacks_total 差值
            var atkDelta = 0;
            if (i > 0) {
                atkDelta = Math.max(0, d.attacks_total - data[i - 1].attacks_total);
            }
            attacksData.push(atkDelta);

            // 系统
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
    } catch (e) { /* silent fail */ }
}

function toggleChart(key, checkbox) {
    if (!timelineChart) return;
    if (key === 'system') {
        // 系统负载 → 切换 CPU(3) + 内存(4) 两条线
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

// ========== 工具函数 ==========

function setText(id, text) {
    var el = document.getElementById(id);
    if (el && el.textContent !== String(text)) {
        el.textContent = text;
    }
}

async function control(action) {
    try {
        var resp = await fetch('/api/control/' + action, { method: 'POST' });
        var data = await resp.json();
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
