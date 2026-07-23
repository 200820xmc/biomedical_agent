@echo off
chcp 65001 >nul
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"
set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"

echo ====================================
echo 启动 AVF Research Assistant
echo ====================================

if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到项目 Python 环境: "%PYTHON_EXE%"
    echo 请先执行: python -m venv .venv
    echo 然后执行: .venv\Scripts\python.exe -m pip install -e .
    exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Docker CLI，请先安装并启动 Docker Desktop。
    exit /b 1
)

echo [1/2] 启动 Milvus 依赖...
docker compose -f "%PROJECT_ROOT%vector-database.yml" up -d etcd minio standalone
if errorlevel 1 (
    echo [错误] Milvus 依赖启动失败。
    exit /b 1
)

echo [2/2] 使用项目 .venv 启动 FastAPI...
echo Web: http://localhost:9900
echo Health: http://localhost:9900/health
"%PYTHON_EXE%" "%PROJECT_ROOT%run_server.py"
exit /b %errorlevel%
