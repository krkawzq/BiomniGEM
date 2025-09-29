import { ChatMessage, ChatSettings, ApiResponse } from '@/types/chat';

export async function sendChatRequest(
  messages: ChatMessage[],
  settings: ChatSettings
): Promise<ApiResponse> {
  const requestBody = {
    messages: messages.map(msg => ({
      role: msg.role,
      content: msg.content
    })),
    base_url: settings.baseUrl,
    api_key: settings.apiKey,
    model: settings.model,
    temperature: settings.temperature || null,
    max_tokens: settings.maxTokens || null,
    top_p: settings.topP || null,
    presence_penalty: settings.presencePenalty || null,
    frequency_penalty: settings.frequencyPenalty || null,
  };

  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API request failed: ${error}`);
  }

  return response.json();
}

export function extractThinking(content: string): { content: string; thinking: string } {
  // 尝试提取 <think>...</think> 标签内容
  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
  
  if (thinkMatch) {
    const thinking = thinkMatch[1].trim();
    const cleanContent = content.replace(/<think>[\s\S]*?<\/think>/i, '').trim();
    return { content: cleanContent, thinking };
  }

  return { content, thinking: '' };
}
