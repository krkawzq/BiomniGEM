import json
import pandas as pd
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_dir", type=str, required=True, help="输出路径")
parser.add_argument("--split_ratio", type=float, required=True, help="train 分割比例")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()

def main():
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    # 设置随机种子并打乱数据
    random.seed(args.seed)
    random.shuffle(data)
    
    train_len = int(len(data) * args.split_ratio)
    train = data[:train_len]
    val = data[train_len:]
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(os.path.join(args.save_dir, "train.json"), "w", encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "val.json"), "w", encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已分割: train={len(train)}, val={len(val)}")

if __name__ == "__main__":
    main()