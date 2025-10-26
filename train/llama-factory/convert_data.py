import json
import datasets
import os

def load_dataset(data_path: str) -> datasets.Dataset:
    dataset = datasets.load_from_disk(data_path)
    return dataset

train_dataset = load_dataset("/root/autodl-fs/wzq/datasets/SynBioCoT/arrow/train")
val_dataset = load_dataset("/root/autodl-fs/wzq/datasets/SynBioCoT/arrow/validation")

train_sharegpt = []
train_sharegpt_sft = []
for item in train_dataset:
    train_sharegpt.append({
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]}
        ]
    })
    if item["quality"] == "gold":
        train_sharegpt_sft.append({
            "messages": [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["assistant"]}
            ]
        })

val_sharegpt = []
for item in val_dataset:
    val_sharegpt.append({
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]}
        ]
    })
    
# ensure path
os.makedirs("/root/autodl-fs/wzq/datasets/SynBioCoT/ShareGPT", exist_ok=True)


with open("/root/autodl-fs/wzq/datasets/SynBioCoT/ShareGPT/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_sharegpt:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("/root/autodl-fs/wzq/datasets/SynBioCoT/ShareGPT/val.jsonl", "w", encoding="utf-8") as f:
    for item in val_sharegpt:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
with open("/root/autodl-fs/wzq/datasets/SynBioCoT/ShareGPT/train_sft.jsonl", "w", encoding="utf-8") as f:
    for item in train_sharegpt_sft:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")