#!/bin/bash
# 启动脚本for FastAPI后端

# 检查是否有uv环境
if [ ! -d ".venv" ]; then
    echo "Creating uv environment..."
    uv venv
fi

# 激活uv环境
source .venv/bin/activate

# 安装依赖
echo "Installing dependencies with uv..."
uv pip install -r requirements.txt

# 启动服务
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
