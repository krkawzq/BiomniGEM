#!/bin/bash
# Chat Box 启动脚本

set -e

echo "🚀 Starting Chat Box..."

# 检查是否在正确的目录
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the ChatBox directory"
    exit 1
fi

echo "📦 Installing dependencies..."

# 后端依赖
echo "Installing backend dependencies..."
cd backend

# 检查是否安装了 uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 创建并激活 uv 环境
if [ ! -d ".venv" ]; then
    echo "Creating uv environment..."
    uv venv
fi

echo "Installing dependencies with uv..."
uv pip install -r requirements.txt
cd ..

# 前端依赖
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "🌟 Starting services..."

# 启动后端服务（后台运行）
echo "Starting backend server on port 8000..."
cd backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# 等待后端启动
sleep 3

# 启动前端服务
echo "Starting frontend server on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Services started successfully!"
echo "🔗 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# 处理终止信号
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 保持脚本运行
wait
