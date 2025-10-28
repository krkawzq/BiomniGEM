import json
import re

# Check training dataset
print("=== Training Dataset ===")
for i in range(3):
    with open('/root/autodl-fs/wzq/datasets/SynBioCoT/jsonl/train.jsonl') as f:
        line = f.readlines()[i]
        data = json.loads(line)
        answer_match = re.search(r'<answer>(.*?)</answer>', data['output'], re.DOTALL)
        print(f"Sample {i}: Has <answer> tag: {answer_match is not None}")
        if not answer_match:
            print(f"  Output ends with: {data['output'][-100:]}")

print("\n=== Validation Dataset ===")
# Check validation dataset  
for i in range(3):
    with open('/root/autodl-fs/wzq/datasets/SynBioCoT/jsonl/cell_validation.jsonl') as f:
        line = f.readlines()[i]
        data = json.loads(line)
        answer_match = re.search(r'<answer>(.*?)</answer>', data['output'], re.DOTALL)
        print(f"Sample {i}: Has <answer> tag: {answer_match is not None}")
        if answer_match:
            print(f"  Answer: {answer_match.group(1).strip()[:50]}")

