'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ChatMessage as ChatMessageType } from '@/types/chat';
import { addLineNumbers } from '@/utils/text';
import { 
  User, 
  Bot, 
  Settings, 
  Plus, 
  Play, 
  Trash2, 
  ChevronDown,
  ChevronRight 
} from 'lucide-react';

interface ChatMessageProps {
  message: ChatMessageType;
  onRoleChange: (id: string, role: 'system' | 'user' | 'assistant') => void;
  onContentChange: (id: string, content: string) => void;
  onDelete: (id: string) => void;
  onGenerate: (afterId: string) => void;
  onAddMessage: (afterId: string) => void;
}

const roleIcons = {
  user: User,
  assistant: Bot,
  system: Settings,
};

const roleColors = {
  user: 'bg-blue-500',
  assistant: 'bg-green-500',
  system: 'bg-purple-500',
};

export default function ChatMessage({
  message,
  onRoleChange,
  onContentChange,
  onDelete,
  onGenerate,
  onAddMessage,
}: ChatMessageProps) {
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);

  const IconComponent = roleIcons[message.role];

  // 同步message.content变化
  React.useEffect(() => {
    setEditContent(message.content);
  }, [message.content]);

  const handleSave = () => {
    onContentChange(message.id, editContent);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditContent(message.content);
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancel();
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSave();
    }
  };

  const numberedText = addLineNumbers(message.content);

  return (
    <div className="border border-gray-200 rounded-lg mb-4 overflow-hidden">
      <div className="flex">
        {/* 左侧角色按钮 */}
        <div className="w-16 bg-gray-50 border-r border-gray-200 flex flex-col items-center py-4">
          <div className="relative group">
            <button
              className={`w-10 h-10 rounded-full ${roleColors[message.role]} flex items-center justify-center text-white hover:opacity-80 transition-opacity`}
              onClick={() => {
                const roles: ('system' | 'user' | 'assistant')[] = ['user', 'assistant', 'system'];
                const currentIndex = roles.indexOf(message.role);
                const nextRole = roles[(currentIndex + 1) % roles.length];
                onRoleChange(message.id, nextRole);
              }}
              title={`切换角色 (当前: ${message.role})`}
            >
              <IconComponent size={20} />
            </button>
            <div className="absolute left-12 top-0 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
              {message.role}
            </div>
          </div>
        </div>

        {/* 双栏内容区域 */}
        <div className="flex-1 flex">
          {/* 左侧：纯文本 + 行号 */}
          <div className="w-1/2 border-r border-gray-200">
            <div className="p-4 h-64 overflow-auto bg-gray-50">
              {isEditing ? (
                <div className="h-full flex flex-col">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="flex-1 w-full p-3 border-2 border-blue-300 rounded font-mono text-sm resize-none bg-white text-gray-900 focus:outline-none focus:border-blue-500 leading-relaxed"
                    placeholder="输入消息内容..."
                    autoFocus
                  />
                  <div className="flex gap-2 mt-2">
                    <button
                      onClick={handleSave}
                      className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition-colors"
                    >
                      保存 (Ctrl+Enter)
                    </button>
                    <button
                      onClick={handleCancel}
                      className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600 transition-colors"
                    >
                      取消 (Esc)
                    </button>
                  </div>
                </div>
              ) : (
                <div 
                  className="relative group cursor-text min-h-full"
                  onClick={() => setIsEditing(true)}
                >
                  <pre className="font-mono text-sm text-gray-900 whitespace-pre-wrap leading-relaxed select-text">
                    {numberedText}
                  </pre>
                  <div className="absolute inset-0 bg-blue-50 opacity-0 group-hover:opacity-30 transition-opacity rounded pointer-events-none" />
                  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                      点击编辑
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 右侧：Markdown渲染 */}
          <div className="w-1/2 bg-white">
            <div className="p-4 h-64 overflow-auto">
              {/* Thinking 过程 */}
              {message.thinking && (
                <div className="mb-4">
                  <button
                    onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
                    className="flex items-center gap-2 text-sm text-gray-800 hover:text-blue-600 font-medium transition-colors"
                  >
                    {isThinkingExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                    思考过程
                  </button>
                  {isThinkingExpanded && (
                    <div className="mt-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                      <div className="text-sm text-gray-800 prose prose-sm max-w-none">
                        <ReactMarkdown
                          components={{
                            code(props) {
                              const { children, className, ...rest } = props;
                              const match = /language-(\w+)/.exec(className || '');
                              return match ? (
                                <SyntaxHighlighter
                                  style={oneDark as any}
                                  language={match[1]}
                                  PreTag="div"
                                  className="text-sm"
                                >
                                  {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                              ) : (
                                <code className="bg-amber-100 text-amber-800 px-1 py-0.5 rounded text-xs" {...rest}>
                                  {children}
                                </code>
                              );
                            },
                          }}
                        >
                          {message.thinking}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 主要内容 */}
              <div className="prose prose-sm max-w-none prose-gray">
                <ReactMarkdown
                  components={{
                    code(props) {
                      const { children, className, ...rest } = props;
                      const match = /language-(\w+)/.exec(className || '');
                      return match ? (
                        <SyntaxHighlighter
                          style={oneDark as any}
                          language={match[1]}
                          PreTag="div"
                          className="text-sm rounded-lg"
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-sm font-mono" {...rest}>
                          {children}
                        </code>
                      );
                    },
                    p: ({ children }) => (
                      <p className="text-gray-900 leading-relaxed mb-3">
                        {children}
                      </p>
                    ),
                    h1: ({ children }) => (
                      <h1 className="text-gray-900 font-bold text-lg mb-3">
                        {children}
                      </h1>
                    ),
                    h2: ({ children }) => (
                      <h2 className="text-gray-900 font-bold text-base mb-2">
                        {children}
                      </h2>
                    ),
                    h3: ({ children }) => (
                      <h3 className="text-gray-900 font-semibold text-sm mb-2">
                        {children}
                      </h3>
                    ),
                    ul: ({ children }) => (
                      <ul className="list-disc list-inside space-y-1 mb-3">
                        {children}
                      </ul>
                    ),
                    ol: ({ children }) => (
                      <ol className="list-decimal list-inside space-y-1 mb-3">
                        {children}
                      </ol>
                    ),
                    li: ({ children }) => (
                      <li className="text-gray-900">
                        {children}
                      </li>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-blue-300 pl-4 italic text-gray-700 bg-blue-50 py-2 rounded-r">
                        {children}
                      </blockquote>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 底部操作按钮 */}
      <div className="border-t border-gray-200 bg-white p-2 flex gap-2">
        <button
          onClick={() => onAddMessage(message.id)}
          className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          <Plus size={14} />
          添加
        </button>
        <button
          onClick={() => onGenerate(message.id)}
          className="flex items-center gap-1 px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
        >
          <Play size={14} />
          生成
        </button>
        <button
          onClick={() => onDelete(message.id)}
          className="flex items-center gap-1 px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
        >
          <Trash2 size={14} />
          删除
        </button>
      </div>
    </div>
  );
}
