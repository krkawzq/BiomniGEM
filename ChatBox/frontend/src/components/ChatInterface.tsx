'use client';

import React, { useState, useCallback } from 'react';
import { ChatMessage as ChatMessageType, ChatSettings } from '@/types/chat';
import { sendChatRequest, extractThinking } from '@/utils/api';
import { generateId } from '@/utils/text';
import ChatMessage from './ChatMessage';
import SettingsPanel from './SettingsPanel';

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessageType[]>([
    {
      id: generateId(),
      role: 'system',
      content: 'You are a helpful assistant.',
      timestamp: Date.now(),
    },
  ]);

  const [settings, setSettings] = useState<ChatSettings>({
    baseUrl: 'https://api.shubiaobiao.cn/v1',
    apiKey: 'sk-Zkut2J4mYzHNZi9752Dc1eE8Ac3d4b48A3A1De46D8EfA4C6',
    model: 'gpt-5-chat-latest',
    temperature: 0,        // 0表示使用模型默认值
    maxTokens: 0,         // 0表示无限制
    topP: 0,              // 0表示使用模型默认值
    presencePenalty: 0,   // 0表示默认值
    frequencyPenalty: 0,  // 0表示默认值
  });

  const [isGenerating, setIsGenerating] = useState(false);

  const handleRoleChange = useCallback((id: string, role: 'system' | 'user' | 'assistant') => {
    setMessages(prev => prev.map(msg => 
      msg.id === id ? { ...msg, role } : msg
    ));
  }, []);

  const handleContentChange = useCallback((id: string, content: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === id ? { ...msg, content } : msg
    ));
  }, []);

  const handleDelete = useCallback((id: string) => {
    setMessages(prev => prev.filter(msg => msg.id !== id));
  }, []);

  const handleAddMessage = useCallback((afterId: string) => {
    const newMessage: ChatMessageType = {
      id: generateId(),
      role: 'user',
      content: '',
      timestamp: Date.now(),
    };

    setMessages(prev => {
      const index = prev.findIndex(msg => msg.id === afterId);
      const newMessages = [...prev];
      newMessages.splice(index + 1, 0, newMessage);
      return newMessages;
    });
  }, []);

  const handleGenerate = useCallback(async (afterId: string) => {
    if (isGenerating) return;

    // 获取当前消息之前的所有消息作为上下文
    const currentIndex = messages.findIndex(msg => msg.id === afterId);

    try {
      setIsGenerating(true);

      const contextMessages = messages.slice(0, currentIndex + 1);

      // 发送API请求
      const response = await sendChatRequest(contextMessages, settings);
      
      // 提取thinking内容
      const { content, thinking } = extractThinking(response.content);

      // 创建新的assistant消息
      const newMessage: ChatMessageType = {
        id: generateId(),
        role: 'assistant',
        content,
        thinking: thinking || undefined,
        timestamp: Date.now(),
      };

      // 删除当前消息之后的所有消息，然后添加新消息
      setMessages(prev => {
        const newMessages = prev.slice(0, currentIndex + 1);
        newMessages.push(newMessage);
        return newMessages;
      });

    } catch (error) {
      console.error('Failed to generate response:', error);
      
      // 创建错误消息
      const errorMessage: ChatMessageType = {
        id: generateId(),
        role: 'assistant',
        content: `❌ **生成失败**\n\n错误信息：${error instanceof Error ? error.message : '未知错误'}\n\n请检查:\n- API 配置是否正确\n- 网络连接是否正常\n- API Key 是否有效`,
        timestamp: Date.now(),
      };

      // 添加错误消息到对话中
      setMessages(prev => {
        const newMessages = prev.slice(0, currentIndex + 1);
        newMessages.push(errorMessage);
        return newMessages;
      });
    } finally {
      setIsGenerating(false);
    }
  }, [messages, settings, isGenerating]);

  const handleSettingsChange = useCallback((newSettings: ChatSettings) => {
    setSettings(newSettings);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-900">
          Chat Box - Prompt 原型设计工具
        </h1>

        {/* 聊天消息列表 */}
        <div className="space-y-4">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onRoleChange={handleRoleChange}
              onContentChange={handleContentChange}
              onDelete={handleDelete}
              onGenerate={handleGenerate}
              onAddMessage={handleAddMessage}
            />
          ))}
        </div>

        {/* 生成状态提示 */}
        {isGenerating && (
          <div className="mt-4 text-center">
            <div className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-800 rounded-lg">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-800 mr-2"></div>
              正在生成回复...
            </div>
          </div>
        )}

        {/* 操作提示 */}
        <div className="mt-8 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
          <h3 className="font-bold mb-4 text-gray-900 text-lg">使用说明：</h3>
          <ul className="text-sm text-gray-800 space-y-2 leading-relaxed">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>点击左侧角色图标切换消息角色（user/assistant/system）</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>点击左侧纯文本区域编辑消息内容，支持快捷键 Ctrl+Enter 保存，Esc 取消</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>点击"添加"按钮在当前位置下方插入新消息</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>点击"生成"按钮基于当前上下文生成AI回复</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>点击"删除"按钮移除当前消息</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 font-bold">•</span>
              <span>右上角设置按钮配置API参数</span>
            </li>
          </ul>
        </div>
      </div>

      {/* 设置面板 */}
      <SettingsPanel
        settings={settings}
        onSettingsChange={handleSettingsChange}
      />
    </div>
  );
}
