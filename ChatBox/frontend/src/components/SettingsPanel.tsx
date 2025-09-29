'use client';

import React, { useState } from 'react';
import { ChatSettings } from '@/types/chat';
import { Settings, X } from 'lucide-react';

interface SettingsPanelProps {
  settings: ChatSettings;
  onSettingsChange: (settings: ChatSettings) => void;
}

export default function SettingsPanel({ settings, onSettingsChange }: SettingsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [tempSettings, setTempSettings] = useState(settings);

  const handleSave = () => {
    onSettingsChange(tempSettings);
    setIsOpen(false);
  };

  const handleCancel = () => {
    setTempSettings(settings);
    setIsOpen(false);
  };

  return (
    <>
      {/* 设置按钮 */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed top-4 right-4 w-12 h-12 bg-gray-800 text-white rounded-full flex items-center justify-center hover:bg-gray-700 transition-colors shadow-lg z-10"
        title="打开设置"
      >
        <Settings size={20} />
      </button>

      {/* 设置面板 */}
      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-[500px] max-w-[90vw] max-h-[90vh] mx-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">API 设置</h2>
              <button
                onClick={handleCancel}
                className="text-gray-500 hover:text-gray-700"
                title="关闭设置"
              >
                <X size={20} />
              </button>
            </div>

            <div className="space-y-5 max-h-96 overflow-y-auto">
              {/* 基础配置 */}
              <div className="space-y-4 border-b border-gray-200 pb-4">
                <h4 className="font-semibold text-gray-800">基础配置</h4>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    Base URL
                  </label>
                  <input
                    type="text"
                    value={tempSettings.baseUrl}
                    onChange={(e) => setTempSettings({ ...tempSettings, baseUrl: e.target.value })}
                    className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors text-gray-900"
                    placeholder="https://api.openai.com/v1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    API Key
                  </label>
                  <input
                    type="password"
                    value={tempSettings.apiKey}
                    onChange={(e) => setTempSettings({ ...tempSettings, apiKey: e.target.value })}
                    className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors text-gray-900"
                    placeholder="sk-..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    模型名称
                  </label>
                  <input
                    type="text"
                    value={tempSettings.model}
                    onChange={(e) => setTempSettings({ ...tempSettings, model: e.target.value })}
                    className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors text-gray-900"
                    placeholder="gpt-4o-mini"
                  />
                  <p className="text-xs text-gray-600 mt-1">输入任何兼容OpenAI API的模型名称</p>
                </div>
              </div>

              {/* 模型参数 */}
              <div className="space-y-4">
                <h4 className="font-semibold text-gray-800">模型参数 <span className="text-xs font-normal text-gray-600">(设为0使用默认值)</span></h4>
                
                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    Temperature: {tempSettings.temperature === 0 ? '默认' : tempSettings.temperature}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={tempSettings.temperature}
                    onChange={(e) => setTempSettings({ ...tempSettings, temperature: parseFloat(e.target.value) })}
                    className="w-full"
                    title="Temperature滑块"
                  />
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>确定性 (0)</span>
                    <span>创造性 (2)</span>
                  </div>
                  <p className="text-xs text-gray-600 mt-1">控制输出的随机性和创造性</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="32000"
                    value={tempSettings.maxTokens}
                    onChange={(e) => setTempSettings({ ...tempSettings, maxTokens: parseInt(e.target.value) || 0 })}
                    className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors text-gray-900"
                    placeholder="0 (无限制)"
                  />
                  <p className="text-xs text-gray-600 mt-1">限制输出的最大令牌数，0表示无限制</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    Top P: {tempSettings.topP === 0 ? '默认' : tempSettings.topP}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={tempSettings.topP}
                    onChange={(e) => setTempSettings({ ...tempSettings, topP: parseFloat(e.target.value) })}
                    className="w-full"
                    title="Top P滑块"
                  />
                  <p className="text-xs text-gray-600 mt-1">控制核心采样，影响词汇选择的多样性</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    Presence Penalty: {tempSettings.presencePenalty === 0 ? '默认' : tempSettings.presencePenalty}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={tempSettings.presencePenalty}
                    onChange={(e) => setTempSettings({ ...tempSettings, presencePenalty: parseFloat(e.target.value) })}
                    className="w-full"
                    title="Presence Penalty滑块"
                  />
                  <p className="text-xs text-gray-600 mt-1">惩罚新主题的出现，促进内容多样性</p>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-800 mb-2">
                    Frequency Penalty: {tempSettings.frequencyPenalty === 0 ? '默认' : tempSettings.frequencyPenalty}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={tempSettings.frequencyPenalty}
                    onChange={(e) => setTempSettings({ ...tempSettings, frequencyPenalty: parseFloat(e.target.value) })}
                    className="w-full"
                    title="Frequency Penalty滑块"
                  />
                  <p className="text-xs text-gray-600 mt-1">惩罚重复词汇，减少重复内容</p>
                </div>
              </div>
            </div>

            <div className="flex gap-3 mt-8">
              <button
                onClick={handleSave}
                className="flex-1 bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                保存设置
              </button>
              <button
                onClick={handleCancel}
                className="flex-1 bg-gray-600 text-white py-3 rounded-lg hover:bg-gray-700 transition-colors font-medium"
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
