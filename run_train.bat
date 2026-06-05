@echo off
REM DS-TCN-IDS 两级训练启动脚本 (v2 重构)
REM
REM 使用方法:
REM   run_train.bat                      → 完整流程：预处理 + 两级训练
REM   run_train.bat train-only           → 跳过预处理，仅训练
REM   run_train.bat preprocess-only      → 仅预处理
REM   run_train.bat medium               → GPU高性能模型

set CONDA_ENV=C:\Users\xunhao\.conda\envs\edge
set PYTHON=%CONDA_ENV%\python.exe
set PROJECT_DIR=I:\毕设\edge-ids

cd /d "%PROJECT_DIR%"

if "%1"=="train-only" (
    echo 🚀 跳过预处理，直接训练...
    %PYTHON% -X utf8 train/train.py --skip-preprocess
) else if "%1"=="preprocess-only" (
    echo 📂 仅预处理数据...
    %PYTHON% -X utf8 data/preprocess.py
) else if "%1"=="medium" (
    echo 🚀 GPU高性能模型训练...
    %PYTHON% -X utf8 train/train.py --model-size medium
) else if "%1"=="tiny" (
    echo 🚀 轻量模型训练 (树莓派5)...
    %PYTHON% -X utf8 train/train.py --model-size tiny
) else (
    echo 🚀 完整训练：预处理 + 两级架构...
    echo.
    echo 📂 Step 0: 数据预处理...
    %PYTHON% -X utf8 data/preprocess.py
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ 预处理失败
        pause
        exit /b 1
    )
    echo.
    echo 🚀 Step 1-2: 两级训练...
    %PYTHON% -X utf8 train/train.py --skip-preprocess
)

pause
