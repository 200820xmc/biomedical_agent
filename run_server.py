"""
启动脚本 —— 为 AVF 科研助手设置代理后启动服务
解决：uvicorn 子进程不继承 PowerShell 环境变量的问题
"""
import os
import sys

# 在导入任何项目模块之前设置代理（代理 = 访问外网的通道）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ['DASHSCOPE_API_BASE'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

print("Proxy set: HTTP_PROXY=127.0.0.1:7890")
print("Starting AVF Research Assistant server...")

# 启动 uvicorn
import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=9900, log_level="info")
