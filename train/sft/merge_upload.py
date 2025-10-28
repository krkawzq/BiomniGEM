#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并 LoRA 适配器到基础模型并上传到 Hugging Face Hub
用法:
    python merge_upload.py --model_path <基础模型路径> --lora_path <LoRA适配器路径> --repo_id <HuggingFace仓库ID> [--private]
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login


def load_base_model(model_path: str):
    """加载基础模型"""
    print(f"加载基础模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model


def load_lora_model(base_model, lora_path: str):
    """加载 LoRA 适配器"""
    print(f"加载 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model


def merge_and_upload(model, tokenizer, repo_id: str, private: bool = False):
    """合并 LoRA 到基础模型并上传到 Hugging Face"""
    print("开始合并 LoRA 适配器...")
    
    # 合并 LoRA 权重到基础模型
    merged_model = model.merge_and_unload()
    
    print(f"上传模型到 Hugging Face Hub: {repo_id}")
    
    # 上传合并后的模型
    merged_model.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Merge LoRA adapter into base model"
    )
    
    # 上传 tokenizer
    tokenizer.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Upload tokenizer"
    )
    
    print("✅ 合并并上传完成!")
    
    # 打印一些信息
    print(f"\n合并后的模型信息:")
    print(f"  - Hugging Face 仓库: {repo_id}")
    print(f"  - 模型参数量: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B")
    print(f"  - 数据类型: {next(merged_model.parameters()).dtype}")
    print(f"  - 访问级别: {'Private' if private else 'Public'}")


def maybe_enable_flash_attn(model):
    """尝试启用 Flash Attention（如果支持）"""
    try:
        model.config.attn_implementation = "flash_attention_2"
        print("✅ Flash Attention 已启用")
    except Exception:
        print("⚠️  Flash Attention 不可用，使用默认实现")
        pass


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器到基础模型并上传到 Hugging Face")
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face 仓库 ID (格式: username/repo-name)")
    parser.add_argument("--private", action="store_true", help="是否创建私有仓库")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token (可选，也可以从环境变量 HF_TOKEN 获取)")
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"基础模型路径不存在: {args.model_path}")
    if not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA 适配器路径不存在: {args.lora_path}")
    
    # 登录 Hugging Face
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        print("使用提供的 token 登录 Hugging Face...")
        login(token=token)
    else:
        print("尝试使用已保存的 Hugging Face 凭证登录...")
        try:
            login()
        except Exception as e:
            print(f"⚠️  登录失败: {e}")
            print("请设置 HF_TOKEN 环境变量或使用 --token 参数提供 API token")
            raise
    
    # 1. 加载基础模型
    base_model = load_base_model(args.model_path)
    
    # 2. 尝试启用 Flash Attention
    maybe_enable_flash_attn(base_model)
    
    # 3. 加载 LoRA 适配器
    lora_model = load_lora_model(base_model, args.lora_path)
    
    # 4. 加载 tokenizer
    print(f"加载 Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 5. 合并并上传
    merge_and_upload(lora_model, tokenizer, args.repo_id, args.private)


if __name__ == "__main__":
    main()
