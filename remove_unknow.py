import json

paths = [
    "/Users/wzq/Documents/Code/Project/BiomniGEM/hf_dataset/text/traces/enhanced_train.json",
    "/Users/wzq/Documents/Code/Project/BiomniGEM/hf_dataset/text/traces/raw_train.json",
    "/Users/wzq/Documents/Code/Project/BiomniGEM/hf_dataset/text/traces/raw_val.json",
    "/Users/wzq/Documents/Code/Project/BiomniGEM/hf_dataset/text/train.json",
    "/Users/wzq/Documents/Code/Project/BiomniGEM/hf_dataset/text/val.json",
    
]

for path in paths:
    with open(path, "r") as f:
        data = json.load(f)

    cleared = []

    for item in data:
        if "Unknown" in item["answer"]:
            pass
        else:
            cleared.append(item)

    with open(path, "w") as f:
        json.dump(cleared, f, ensure_ascii=False, indent=2)