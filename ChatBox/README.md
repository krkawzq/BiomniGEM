# Chat Box - Prompt 原型设计工具

一个轻量化的 Prompt 原型设计工具，支持快速测试不同提示词策略，并能直观对比模型原始输出与渲染结果。

## 功能特点

### 🎯 核心功能
- **双栏显示**：左侧显示带行号的纯文本，右侧显示 Markdown 渲染结果
- **对话管理**：支持角色切换（system/user/assistant）、编辑、删除、插入
- **Thinking 展开**：支持查看 AI 的思考过程（如 `<think>...</think>` 标签内容）
- **原地编辑**：点击纯文本区域即可原地编辑，支持快捷键操作
- **灵活配置**：支持自定义 BaseURL、API Key 和任意模型名称
- **精细调优**：完整的模型参数控制（温度、最大tokens、top_p、惩罚系数等）
- **错误处理**：友好的错误提示和自动重试机制

### ✨ 用户体验优化
- **高对比度**：优化文字颜色，提升可读性
- **快捷键支持**：Ctrl+Enter 保存，Esc 取消编辑
- **悬停提示**：鼠标悬停显示操作提示
- **实时反馈**：编辑状态和加载状态的清晰提示
- **参数可视化**：直观的滑块控制和实时数值显示

### 🛠 技术栈
- **前端**：Next.js + TypeScript + Tailwind CSS
- **后端**：FastAPI + LangChain + OpenAI
- **UI 组件**：Lucide React Icons
- **Markdown 渲染**：react-markdown + react-syntax-highlighter

## 快速开始

### 前置要求
- Node.js 18+
- Python 3.8+
- npm 或 yarn
- uv (Python包管理器)

### 一键启动
```bash
# 给启动脚本执行权限
chmod +x start.sh

# 运行启动脚本
./start.sh
```

启动脚本会自动：
1. 安装后端和前端依赖
2. 启动后端服务 (端口 8000)
3. 启动前端服务 (端口 3000)

### 手动启动

#### 后端服务
```bash
# 如果还没有安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 前端服务
```bash
cd frontend
npm install
npm run dev
```

### 访问地址
- **前端界面**：http://localhost:3000
- **后端 API**：http://localhost:8000
- **API 文档**：http://localhost:8000/docs

## 使用说明

### 1. 配置 API
1. 点击右上角的设置按钮
2. 输入你的 API Base URL 和 API Key
3. 选择要使用的模型
4. 点击保存

### 2. 使用界面
- **角色切换**：点击左侧圆形图标可在 user/assistant/system 角色间切换
- **编辑内容**：点击左侧纯文本区域进入编辑模式，支持快捷键操作
- **查看渲染**：右侧自动显示 Markdown 渲染结果
- **展开思考**：如果 AI 返回包含 thinking 内容，可点击展开查看
- **参数调节**：通过滑块调整温度、top_p等模型参数，0值表示使用默认

### 3. 对话操作
- **添加消息**：点击"添加"按钮在当前位置下方插入新的空白消息
- **生成回复**：点击"生成"按钮基于当前上下文让 AI 生成回复
- **删除消息**：点击"删除"按钮移除当前消息

### 4. Thinking 过程
工具会自动检测和提取 `<think>...</think>` 标签内的内容作为 AI 的思考过程，可以在右侧面板中展开查看。

## 项目结构

```
ChatBox/
├── backend/                 # Python FastAPI 后端
│   ├── main.py             # 主应用文件
│   ├── requirements.txt    # Python 依赖
│   └── start.sh           # 后端启动脚本
├── frontend/               # Next.js 前端
│   ├── src/
│   │   ├── app/           # Next.js 应用目录
│   │   ├── components/    # React 组件
│   │   ├── types/         # TypeScript 类型定义
│   │   └── utils/         # 工具函数
│   ├── package.json       # 前端依赖
│   └── ...
├── start.sh               # 一键启动脚本
└── README.md             # 项目说明
```

## API 接口

### POST /api/chat
发送聊天请求到 AI 模型

**请求体：**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model": "gpt-4o-mini"
}
```

**响应：**
```json
{
  "content": "Hello! How can I help you today?",
  "raw_response": {
    "content": "Hello! How can I help you today?",
    "response_metadata": {},
    "usage_metadata": {}
  }
}
```

### GET /api/health
健康检查接口

## 开发说明

### 后端开发
- 基于 FastAPI 框架
- 使用 LangChain 集成各种 LLM
- 支持 OpenAI 兼容的 API 接口
- 自动提取 thinking 内容

### 前端开发
- 基于 Next.js 14 App Router
- 使用 TypeScript 确保类型安全
- Tailwind CSS 负责样式
- 响应式设计适配不同屏幕

### 环境要求
- Python 3.8+（推荐 3.10+）
- Node.js 18+（推荐 20+）
- uv（Python包管理器，比pip更快）

### 安装 uv
如果你还没有安装 uv，可以运行：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目遵循 MIT 许可证。
