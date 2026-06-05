@echo off
REM DS-TCN-IDS 训练启动脚本
REM 使用方法: 双击运行 或 命令行传参
REM    run_train.bat            → 默认 small 模型
REM    run_train.bat tiny       → 轻量模型
REM    run_train.bat ablation   → 消融实验

set CONDA_ENV=C:\Users\xunhao\.conda\envs\edge
set PYTHON=%CONDA_ENV%\python.exe

cd /d "I:\毕设\edge-ids"

if "%1"=="tiny" (
    echo 🚀 启动 tiny 模型训练 (树莓派5部署)...
    %PYTHON% train/train.py --model-size tiny --epochs 50 --batch-size 64
) else if "%1"=="medium" (
    echo 🚀 启动 medium 模型训练 (GPU高性能)...
    %PYTHON% train/train.py --model-size medium --epochs 50 --batch-size 128
) else if "%1"=="ablation" (
    echo 🔬 启动消融实验...
    %PYTHON% train/train.py --ablation --epochs 50
) else if "%1"=="preprocess" (
    echo 📂 数据预处理...
    %PYTHON% data/preprocess.py
) else (
    echo 🚀 启动 small 模型训练 (推荐)...
    %PYTHON% train/train.py --model-size small --epochs 50 --batch-size 64
)

pause
