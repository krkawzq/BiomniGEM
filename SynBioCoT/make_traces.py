import pandas as pd
import argparse
import os
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--task", type=str, required=True, help="任务名称")
args = parser.parse_args()

def main():
    df = pd.read_csv(args.data_path)
    new_df = pd.DataFrame()
    new_df["system"] = df["system"]
    new_df["user"] = df["task"]
    new_df["response"] = df["response"]
    new_df["answer"] = df["answer"]
    
    def get_traces(response: str):
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                traces = data.get("traces", [])
                return traces
            return pd.NA
        except Exception as e:
            return pd.NA
    
    new_df["traces"] = new_df["response"].apply(get_traces)
    new_df = new_df[new_df["traces"].notna()].dropna()
    
    datas = []
    
    for _, row in new_df.iterrows():
        datas.append({
            "system": row["system"],
            "user": row["user"],
            "traces": row["traces"],
            "answer": row["answer"],
            "task": args.task,
        })
    
    # 保存结果到指定路径
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(datas)} 条数据，已保存到 {args.save_path}")

if __name__ == "__main__":
    main()