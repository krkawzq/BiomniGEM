import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, nargs="+", required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
args = parser.parse_args()

def main():
    data = []
    for path in args.data_path:
        with open(path, "r") as f:
            data.extend(json.load(f))
    json.dump(data, open(args.save_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items to {args.save_path}")

if __name__ == "__main__":
    main()