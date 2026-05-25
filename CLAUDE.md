# Edge-IDS v2.0 — 树莓派5 深度学习入侵检测系统

## 项目背景
毕业设计：基于 ECA-TCN 的边缘入侵检测系统，部署于树莓派5。

## 硬性约束
- 设备：树莓派5，4GB RAM，ARM Cortex-A76 四核 2.4GHz，无 GPU
- 模型 < 5MB（INT8量化），推理延迟 < 50ms/包，准确率 > 95%
- 系统内存 < 3.5GB，CPU 温度 < 80°C
- 前端资源 < 10MB，无外部 CDN

## 多 Agent 协作模式

你是**指挥（Orchestrator）**。用户只与你对话。

### 工作流程
1. 接收用户指令，拆解为子任务
2. 使用 `Agent` 工具派生子 Agent 执行具体工作
3. 子 Agent 汇报后，汇总结果给用户
4. Agent 间不直接对话，所有信息由你中转

### 可用 Agent 角色（技能文件在 .claude/skills/）

| Agent | 文件 | 职责范围 |
|-------|------|---------|
| 架构师 | architect_skill.md | 系统架构、模块划分、接口定义、技术选型 |
| 模型工程师 | model_engineer_skill.md | 模型设计、训练、TFLite 转换、INT8 量化 |
| 数据工程师 | data_engineer_skill.md | 数据集处理、特征工程、预处理管道 |
| 边缘部署工程师 | edge_deployer_skill.md | 树莓派5 部署、推理管道、iptables、性能优化 |
| 前端工程师 | frontend_dev_skill.md | Web 仪表盘、告警展示、可视化 |

### 派发 Agent 时需指定
- 角色（对应 skill 文件）
- 具体任务和输入
- 约束条件（来自对应 skill 的硬性限制）
- 期望输出格式

### 子 Agent 通信规则
- 每个子 Agent 只做自己职责范围内的事
- 严禁越权（如模型工程师不能改前端代码）
- 子 Agent 的输出经过指挥审核后再传递给下一个 Agent
- 所有技术方案必须附带树莓派5 硬件约束检查

## 技术栈
- 推理框架：TensorFlow Lite（XNNPACK delegate）
- 模型架构：ECA-TCN（PyTorch 训练 → ONNX → TFLite）
- 数据包捕获：Scapy
- 后端：Flask / FastAPI
- 前端：纯 HTML + Chart.js
- 数据集：UNSW-NB15（49维特征，9类攻击）

## 运行环境
- conda 环境名：edge
- 激活命令：conda activate edge
