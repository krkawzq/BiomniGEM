import pandas as pd
import argparse
import os
from cellm.task import EvalTask
from cellm.process.template import Template

sc_ann_task_abstract = "You now need to determine the most probable cell type of a single cell based on its expression profile, which is represented by the top highly expressed genes sorted in descending order."

sc_ann_task_description_list = [
    "Given the expression profile of a single cell, identify its most probable cell type.",
    "You will receive a list of genes with the highest expression in one single cell. These highly expressed genes serve as a proxy for cell identity and can be used to infer the most likely cell type.",
    "The gene list is obtained from **single-cell RNA sequencing**, with genes sorted in descending order of expression (higher first, lower later)."
]

system_base = """You are a helpful biology expert. {task_abstract}

## Task
{task_description_list}
"""

user_base = """{task}

You are given the correct answer:
{answer}

---

Please explain **why** this answer is correct by providing a clear and detailed reasoning process
before stating the final conclusion.

Follow these requirements carefully:
- You must reason using biological knowledge and domain understanding.
- Each reasoning step must have clear biological meaning and causal coherence.
- You must not reveal or imply that you already know the final answer; reasoning should unfold naturally.
- Each step should logically follow from the previous one, forming a consistent causal chain.
- Present your final reasoning and conclusion in a well-formatted JSON block as shown below:

```json
{{
    "traces": [
        "...", 
        "...", 
        "..."
    ],
    "conclusion": "..."
}}
"""


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--task", type=str, required=True, help="任务名称")
parser.add_argument("--samples", type=int, required=True, help="样本数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()


def main():
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
        "task_abstract": sc_ann_task_abstract, 
        "task_description_list": sc_ann_task_description_list
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