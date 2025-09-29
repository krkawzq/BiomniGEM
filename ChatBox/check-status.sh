#!/bin/bash
# 检查 Chat Box 服务状态

echo "🔍 检查 Chat Box 服务状态..."

# 检查后端服务
echo "📡 检查后端服务..."
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ 后端服务正常运行 (http://localhost:8000)"
    # 获取健康状态
    HEALTH=$(curl -s http://localhost:8000/api/health)
    echo "   状态: $HEALTH"
else
    echo "❌ 后端服务未运行或无法访问"
fi

echo ""

# 检查前端服务
echo "🌐 检查前端服务..."
if curl -s -I http://localhost:3000 | grep -q "200 OK"; then
    echo "✅ 前端服务正常运行 (http://localhost:3000)"
else
    echo "❌ 前端服务未运行或无法访问"
fi

echo ""

# 检查进程
echo "🔧 检查相关进程..."
BACKEND_PID=$(pgrep -f "uvicorn main:app")
FRONTEND_PID=$(pgrep -f "next dev")

if [ ! -z "$BACKEND_PID" ]; then
    echo "✅ 后端进程运行中 (PID: $BACKEND_PID)"
else
    echo "❌ 后端进程未找到"
fi

if [ ! -z "$FRONTEND_PID" ]; then
    echo "✅ 前端进程运行中 (PID: $FRONTEND_PID)"
else
    echo "❌ 前端进程未找到"
fi

echo ""
echo "🎯 如果所有服务都正常运行，请访问："
echo "   前端界面: http://localhost:3000"
echo "   API文档:  http://localhost:8000/docs"
