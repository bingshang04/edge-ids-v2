@echo off
REM Edge-IDS v2.0 启动脚本 (Windows)
cd /d "%~dp0"

set MODE=full
set INTERFACE=
set CONFIG=config.yaml

:parse
if "%1"=="" goto run
if "%1"=="--mode" set MODE=%2& shift & shift & goto parse
if "%1"=="--interface" set INTERFACE=%2& shift & shift & goto parse
if "%1"=="--config" set CONFIG=%2& shift & shift & goto parse
if "%1"=="--help" goto help
echo 未知选项: %1
goto end

:help
echo Edge-IDS v2.0 (ECA-TCN)
echo 用法: run.bat [--mode full^|capture^|dashboard] [--interface WLAN] [--config config.yaml]
echo 注意: 数据包捕获需要管理员权限运行
goto end

:run
echo Edge-IDS v2.0 启动 ^| 模式: %MODE% ^| 配置: %CONFIG%
if "%INTERFACE%"=="" (
    python main.py --mode %MODE% --config %CONFIG%
) else (
    python main.py --mode %MODE% --interface %INTERFACE% --config %CONFIG%
)

:end
pause
