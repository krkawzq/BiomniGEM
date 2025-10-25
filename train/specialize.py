import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

def main():
    parser = argparse.ArgumentParser(description="检查并添加特殊token到模型词表")
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument("--save_path", type=str, required=True, help="保存新模型路径")
    parser.add_argument("--special_tokens", type=str, nargs="+", required=True, help="要添加的特殊token列表")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print("=" * 80)
    print("🚀 开始处理模型特殊 token")
    print("=" * 80)
    print(f"📂 原始模型路径: {args.model_path}")
    print(f"💾 保存路径: {args.save_path}")
    print(f"🔤 待检查的特殊 token: {args.special_tokens}")
    print(f"🎲 随机种子: {args.seed}")
    print("=" * 80)

    print(f"\n🔹 加载模型和 tokenizer：{args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"✅ 模型加载完成")
    print(f"📊 当前词表大小: {len(tokenizer)}")
    print(f"📊 模型参数量: {model.num_parameters():,}")

    # 检查哪些 token 缺失
    print("\n" + "=" * 80)
    print("🔍 检查特殊 token 是否存在于词表中...")
    print("=" * 80)
    missing_tokens = []
    existing_tokens = []
    for tok in args.special_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id == tokenizer.unk_token_id:
            print(f"❌ 未找到：{tok} (将被添加)")
            missing_tokens.append(tok)
        else:
            print(f"✅ 已存在：{tok} (ID={tok_id})")
            existing_tokens.append((tok, tok_id))

    print("\n" + "-" * 80)
    print(f"📈 统计结果:")
    print(f"   ✅ 已存在的 token 数量: {len(existing_tokens)}")
    print(f"   ❌ 缺失的 token 数量: {len(missing_tokens)}")
    print("-" * 80)

    # 若有缺失则添加
    if missing_tokens:
        print(f"\n🚀 开始添加新的特殊 token：{missing_tokens}")
        original_vocab_size = len(tokenizer)
        tokenizer.add_special_tokens({'additional_special_tokens': missing_tokens})
        new_vocab_size = len(tokenizer)
        print(f"📊 词表大小变化: {original_vocab_size} → {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
        
        print(f"🔧 调整模型 embedding 层大小...")
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Embedding 层调整完成")
        
        # 验证新添加的 token
        print(f"\n🔍 验证新添加的 token:")
        for tok in missing_tokens:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            print(f"   ✅ {tok} → ID={tok_id}")
    else:
        print("\n✅ 无需添加新 token，词表已完整。")

    # 保存扩展后的模型与 tokenizer
    print("\n" + "=" * 80)
    print(f"💾 保存模型到 {args.save_path}")
    print("=" * 80)
    os.makedirs(args.save_path, exist_ok=True)
    
    print("💾 保存 tokenizer...")
    tokenizer.save_pretrained(args.save_path)
    print("✅ Tokenizer 保存完成")
    
    print("💾 保存模型...")
    model.save_pretrained(args.save_path)
    print("✅ 模型保存完成")

    print("\n" + "=" * 80)
    print("🎉 处理完成！")
    print("=" * 80)
    print(f"📂 原始模型: {args.model_path}")
    print(f"💾 新模型保存到: {args.save_path}")
    print(f"📊 最终词表大小: {len(tokenizer)}")
    print(f"📊 特殊 token 总数: {len(args.special_tokens)}")
    print(f"   - 原本存在: {len(existing_tokens)}")
    print(f"   - 新增加的: {len(missing_tokens)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
