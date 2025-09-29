#!/bin/bash
# 测试项目设置脚本

echo "🧪 Testing Chat Box setup..."

# 检查文件结构
echo "📂 Checking file structure..."
if [ ! -f "start.sh" ]; then
    echo "❌ Missing start.sh"
    exit 1
fi

if [ ! -d "backend" ]; then
    echo "❌ Missing backend directory"
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo "❌ Missing frontend directory"
    exit 1
fi

if [ ! -f "backend/requirements.txt" ]; then
    echo "❌ Missing backend/requirements.txt"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo "❌ Missing frontend/package.json"
    exit 1
fi

echo "✅ File structure looks good"

# 检查 Python
echo "🐍 Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
echo "✅ Python3 found: $(python3 --version)"

# 检查 Node.js
echo "📦 Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    exit 1
fi
echo "✅ Node.js found: $(node --version)"

# 检查 npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found"
    exit 1
fi
echo "✅ npm found: $(npm --version)"

# 检查 uv
echo "🚀 Checking uv..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✅ uv found: $(uv --version)"

echo ""
echo "🎉 Setup test completed successfully!"
echo "✨ You can now run './start.sh' to launch the application"
