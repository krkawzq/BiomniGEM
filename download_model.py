#!/usr/bin/env python3
"""
从 Hugging Face 下载 LLM 模型的简单脚本
使用方法: python download_model.py --model <model_name> --output <output_dir>
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name, output_dir):
    """
    下载指定的 LLM 模型和 tokenizer
    
    Args:
        model_name: Hugging Face 模型名称，如 "gpt2", "meta-llama/Llama-2-7b-hf"
        output_dir: 模型保存路径
    """
    print(f"开始下载模型: {model_name}")
    print(f"保存位置: {output_dir}")
    
    try:
        # 下载 tokenizer
        print("\n[1/2] 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=output_dir,
            trust_remote_code=True
        )
        print("✓ Tokenizer 下载完成")
        
        # 下载模型
        print("\n[2/2] 下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=output_dir,
            trust_remote_code=True
        )
        print("✓ 模型下载完成")
        
        print(f"\n✓ 全部完成！模型已保存到: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {str(e)}")
        print("\n提示:")
        print("- 如果是私有模型，需要先登录: huggingface-cli login")
        print("- 检查模型名称是否正确")
        print("- 确保有足够的磁盘空间")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 下载 LLM 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_model.py --model gpt2 --output ./models
  python download_model.py --model meta-llama/Llama-2-7b-hf --output /data/models
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face 模型名称，如: gpt2, Qwen/Qwen-7B"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="模型保存路径"
    )
    
    args = parser.parse_args()
    
    download_model(args.model, args.output)


if __name__ == "__main__":
    main()

