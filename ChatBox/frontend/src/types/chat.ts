export interface ChatMessage {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp: number;
  thinking?: string; // AI的思考过程
}

export interface ChatSettings {
  baseUrl: string;
  apiKey: string;
  model: string;
  temperature: number;  // 0为默认，由模型决定
  maxTokens: number;    // 0为默认，无限制
  topP: number;         // 0为默认，由模型决定
  presencePenalty: number;  // 0为默认
  frequencyPenalty: number; // 0为默认
}

export interface ApiResponse {
  content: string;
  raw_response: {
    content: string;
    response_metadata?: any;
    usage_metadata?: any;
  };
}
