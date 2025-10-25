import pandas as pd
import argparse
import os
from cellm.task import EvalTask
from cellm.dataset import Dataset
from openai import AsyncOpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--base_url", type=str, default=None, help="基础URL")
parser.add_argument("--api_key", type=str, default=None, help="API密钥")
parser.add_argument("--model", type=str, required=True, help="模型名称")
parser.add_argument("--temperature", type=float, default=None, help="温度参数")
parser.add_argument("--top_p", type=float, default=None, help="Top-p参数")
parser.add_argument("--max_concurrency", type=int, default=5, help="最大并发数")
parser.add_argument("--samples", type=int, default=None, help="样本数量")
args = parser.parse_args()

def main():
    # -- validate and ensure save path
    if not args.save_path.endswith(".csv"):
        print(f"Warning: Save path {args.save_path} is not a csv file")
        exit(1)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # -- load API credentials
    if args.base_url is None or args.api_key is None:
        print("Loading API credentials from .env file...")
        print("--------------------------------")
        if not os.path.exists(".env"):
            print("Error: .env file not found")
            exit(1)
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("BASE_URL="):
                    args.base_url = line.split("=")[1].strip()
                elif line.startswith("API_KEY="):
                    args.api_key = line.split("=")[1].strip()
        if args.base_url is None or args.api_key is None:
            print("Error: BASE_URL or API_KEY not found in .env file")
            exit(1)
    
    # -- initialize client and task
    print("Initializing OpenAI client...")
    print("--------------------------------")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    task = EvalTask(client=client)
    
    # -- load data
    print(f"Loading data from {args.data_path}")
    print("--------------------------------")
    df = pd.read_csv(args.data_path)
    dataset = Dataset(df)
    
    # -- run task
    print(f"Running task with model {args.model}...")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Samples: {args.samples if args.samples else 'all'}")
    print("--------------------------------")
    result = task.run(
        model=args.model,
        dataset=dataset,
        temperature=args.temperature,
        top_p=args.top_p,
        max_concurrency=args.max_concurrency,
        samples=args.samples,
    )
    
    # -- save result
    print(f"Saving to {args.save_path}...")
    print("--------------------------------")
    result.to_csv(args.save_path, index=False)
    print("--------------------------------")
    print("all done!")

if __name__ == "__main__":
    main()