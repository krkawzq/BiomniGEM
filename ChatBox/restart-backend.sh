#!/bin/bash
# 快速重启后端服务脚本

echo "🔄 重启后端服务..."

# 杀死可能正在运行的后端进程
echo "停止现有后端进程..."
pkill -f "uvicorn main:app" || true

# 等待进程完全停止
sleep 2

# 启动新的后端服务
echo "启动后端服务..."
cd backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

echo "✅ 后端服务已重启"
echo "🔗 API地址: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"
