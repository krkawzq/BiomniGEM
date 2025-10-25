import pandas as pd
import argparse
import os
from cellm.task import EvalTask
from cellm.process.template import Template
from prompt.tasks import *
from prompt.system import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--task", type=str, required=True, help="任务名称")
parser.add_argument("--samples", type=int, required=True, help="样本数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()

all_tasks = {}
for key, value in tasks.items():
    all_tasks[key] = value

def main():
    # -- validate task
    if not args.task in tasks:
        print(f"Error: Task {args.task} not found")
        exit(1)
    
    # -- ensure save path
    if not args.save_path.endswith(".csv"):
        print(f"Warning: Save path {args.save_path} is not a csv file")
        exit(1)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # -- load data
    print("--------------------------------")
    print(f"Loading data from {args.data_path}")
    print("--------------------------------")
    df = pd.read_csv(args.data_path)
    
    # -- apply template
    print("Applying template...")
    print("--------------------------------")
    template = Template({
        "system": system_base,
        "user": user_base,
        "task": "{task}",
        "answer": "{answer}"
    })
    df = template.apply(df, appends={
        "task_abstract": tasks[args.task]["task_abstract"], 
        "task_description_list": tasks[args.task]["task_description_list"]
    })
    
    # -- random sampling
    if args.samples is not None and args.samples < len(df):
        print(f"Randomly sampling {args.samples} rows from {len(df)} total rows...")
        print("--------------------------------")
        df = df.sample(n=args.samples, random_state=args.seed)
        df = df.reset_index(drop=True)
    
    # -- save result
    print(f"Saving to {args.save_path}...")
    print("--------------------------------")
    df.to_csv(args.save_path, index=False)
    print("--------------------------------")
    print("all done!")

if __name__ == "__main__":
    main()