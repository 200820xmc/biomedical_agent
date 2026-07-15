@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ====================================
echo 启动 AVF 科研助手
echo ====================================
echo.

REM 检查 uv 是否安装
echo [1/4] 检查包管理器...
where uv >nul 2>&1
if errorlevel 1 (
    echo [信息] uv 未安装，将使用传统 pip 方式
    set USE_UV=0
) else (
    echo [成功] 检测到 uv 包管理器
    set USE_UV=1
)
echo.

REM 确保 Python 版本正确
echo [2/4] 配置 Python 版本...
if exist .python-version (
    set /p PYTHON_VERSION=<.python-version
    echo [信息] 当前配置版本: !PYTHON_VERSION!
) else (
    echo [信息] 创建 .python-version 文件...
    echo 3.13> .python-version
)
echo.

REM 启动 Docker Compose
echo [3/4] 启动 Milvus 向量数据库...
docker ps --format "{{.Names}}" | findstr "milvus-standalone" >nul 2>&1
if not errorlevel 1 (
    echo [信息] Milvus 容器已在运行
) else (
    docker compose -f vector-database.yml up -d etcd minio standalone
    if errorlevel 1 (
        echo [错误] Docker 启动失败，请确保 Docker Desktop 已启动
        pause
        exit /b 1
    )
    echo [信息] 等待 Milvus 启动（10秒）...
    timeout /t 10 /nobreak >nul
)
echo [成功] Milvus 数据库就绪
echo.

REM 启动 FastAPI 服务
echo [4/4] 启动 FastAPI 服务...
start "AVF Research Assistant" python run_server.py
echo [信息] 等待服务启动（15秒）...
timeout /t 15 /nobreak >nul
echo.

REM 检查服务状态
echo [信息] 检查服务状态...
curl -s http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [警告] 服务可能还未完全启动，请稍等片刻
) else (
    echo [成功] FastAPI 服务运行正常
)

echo.
echo ====================================
echo AVF 科研助手启动完成！
echo ====================================
echo Web 界面: http://localhost:9900
echo API 文档: http://localhost:9900/docs
echo 停止服务: stop-windows.bat
echo ====================================
pause
