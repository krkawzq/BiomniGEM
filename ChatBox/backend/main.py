from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

app = FastAPI(title="Chat Box API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.66.184.91:3000",  # 支持网络访问
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 请求模型
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    base_url: str
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: Optional[float] = None  # None表示使用默认值
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

class ChatResponse(BaseModel):
    content: str
    raw_response: Dict[str, Any]

# 全局变量存储LLM客户端
llm_client = None

def get_llm_client(base_url: str, api_key: str, model: str, 
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    top_p: Optional[float] = None,
                    presence_penalty: Optional[float] = None,
                    frequency_penalty: Optional[float] = None):
    """获取或创建LLM客户端"""
    global llm_client
    try:
        # 构建参数字典，只包含非None的值
        kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }
        
        # 添加可选参数，0值表示使用默认值（不传递给API）
        if temperature is not None and temperature != 0:
            kwargs["temperature"] = temperature
        if max_tokens is not None and max_tokens != 0:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None and top_p != 0:
            kwargs["top_p"] = top_p
        if presence_penalty is not None and presence_penalty != 0:
            kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None and frequency_penalty != 0:
            kwargs["frequency_penalty"] = frequency_penalty
            
        llm_client = ChatOpenAI(**kwargs)
        return llm_client
    except Exception as e:
        error_msg = str(e)
        # 提供更友好的错误信息
        if "socksio" in error_msg:
            error_msg = "网络代理配置问题，请检查网络设置"
        elif "api" in error_msg.lower() and "key" in error_msg.lower():
            error_msg = "API Key 无效或已过期"
        elif "base_url" in error_msg.lower() or "url" in error_msg.lower():
            error_msg = "Base URL 配置错误，请检查API地址"
        elif "model" in error_msg.lower():
            error_msg = f"模型 '{model}' 不支持或不存在"
        else:
            error_msg = f"LLM客户端初始化失败: {error_msg}"
        
        raise HTTPException(status_code=400, detail=error_msg)

def convert_messages(messages: List[ChatMessage]):
    """转换消息格式为LangChain格式"""
    langchain_messages = []
    for msg in messages:
        if msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    return langchain_messages

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    try:
        # 获取LLM客户端
        llm = get_llm_client(
            request.base_url, 
            request.api_key, 
            request.model,
            request.temperature,
            request.max_tokens,
            request.top_p,
            request.presence_penalty,
            request.frequency_penalty
        )
        
        # 转换消息格式
        messages = convert_messages(request.messages)
        
        # 调用LLM
        response = llm.invoke(messages)
        
        # 构造响应
        raw_response = {
            "content": response.content,
            "response_metadata": getattr(response, 'response_metadata', {}),
            "usage_metadata": getattr(response, 'usage_metadata', {})
        }
        
        return ChatResponse(
            content=response.content,
            raw_response=raw_response
        )
        
    except Exception as e:
        error_msg = str(e)
        # 提供更友好的错误信息
        if "rate limit" in error_msg.lower():
            error_msg = "API请求频率超限，请稍后重试"
        elif "quota" in error_msg.lower():
            error_msg = "API配额不足，请检查账户余额"
        elif "invalid api key" in error_msg.lower():
            error_msg = "API Key 无效，请检查配置"
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            error_msg = "指定的模型不存在或无权限访问"
        elif "timeout" in error_msg.lower():
            error_msg = "请求超时，请检查网络连接"
        else:
            error_msg = f"聊天请求失败: {error_msg}"
            
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "message": "Chat Box API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
