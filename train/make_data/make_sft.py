import json
import argparse
import os
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--only_format", type=str, default=None, choices=["flex", "fixed"], help="只生成指定格式的数据")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()

answer_bases = [
    "The correct answer is {answer}.",
    "The answer is {answer}.",
    "Based on the analysis, the answer is {answer}.",
    "After careful consideration, I conclude that {answer}.",
    "The result is {answer}.",
    "My prediction is {answer}.",
    "According to the data, the answer is {answer}.",
    "The final answer is {answer}.",
    "I believe the answer is {answer}.",
    "The most likely answer is {answer}.",
    "Based on my analysis, {answer}.",
    "The evidence suggests that {answer}.",
    "After evaluating all factors, {answer}.",
    "My conclusion is {answer}.",
    "The predicted result is {answer}.",
    "I would say the answer is {answer}.",
    "The outcome is {answer}.",
    "Based on the reasoning above, {answer}.",
    "Therefore, the answer is {answer}.",
    "In conclusion, {answer}.",
    "The analysis indicates that {answer}.",
    "My assessment is {answer}.",
    "The data shows that {answer}.",
    "I conclude that {answer}.",
    "The answer appears to be {answer}.",
    "Based on these observations, {answer}.",
    "The final result is {answer}.",
    "After thorough analysis, {answer}.",
    "The prediction yields {answer}.",
    "I determine that {answer}.",
    "The calculated result is {answer}.",
    "Based on the evidence, {answer}.",
    "My final answer is {answer}.",
    "The solution is {answer}.",
    "I predict that {answer}.",
    "The answer should be {answer}.",
    "Based on this reasoning, {answer}.",
    "The expected result is {answer}.",
    "I estimate that {answer}.",
    "The findings suggest {answer}.",
    "After consideration, the answer is {answer}.",
    "The most probable answer is {answer}.",
    "I would conclude that {answer}.",
    "The result appears to be {answer}.",
    "Based on the information, {answer}.",
    "My determination is {answer}.",
    "The answer would be {answer}.",
    "I find that {answer}.",
    "The outcome appears to be {answer}.",
    "Based on my evaluation, {answer}.",
    "The predicted outcome is {answer}.",
    "I assess that {answer}.",
    "The answer seems to be {answer}.",
    "After analyzing the data, {answer}.",
    "The conclusion is {answer}.",
    "I infer that {answer}.",
    "The result should be {answer}.",
    "Based on the pattern, {answer}.",
    "My inference is {answer}.",
    "The answer is likely {answer}.",
    "I deduce that {answer}.",
    "The expected answer is {answer}.",
    "Based on the model, {answer}.",
    "My prediction yields {answer}.",
    "The answer must be {answer}.",
    "I reason that {answer}.",
    "The final prediction is {answer}.",
    "Based on the calculation, {answer}.",
    "My analysis suggests {answer}.",
    "The answer can be determined as {answer}.",
    "I hypothesize that {answer}.",
    "The result indicates {answer}.",
    "Based on the features, {answer}.",
    "My hypothesis is {answer}.",
    "The answer is determined to be {answer}.",
    "I surmise that {answer}.",
    "The predicted value is {answer}.",
    "Based on the sequence, {answer}.",
    "My estimation is {answer}.",
    "The answer is predicted as {answer}.",
    "I postulate that {answer}.",
    "The final outcome is {answer}.",
    "Based on the structure, {answer}.",
    "My conclusion suggests {answer}.",
    "The answer is estimated as {answer}.",
    "I theorize that {answer}.",
    "The result can be {answer}.",
    "Based on the characteristics, {answer}.",
    "My theory is {answer}.",
    "The answer is calculated as {answer}.",
    "I speculate that {answer}.",
    "The predicted answer is {answer}.",
    "Based on the properties, {answer}.",
    "My speculation is {answer}.",
    "The answer is inferred as {answer}.",
    "I conjecture that {answer}.",
    "The final value is {answer}.",
    "Based on the composition, {answer}.",
    "My conjecture is {answer}.",
    "The answer is deduced as {answer}.",
    "So the answer is {answer}.",
    "Thus, {answer}.",
    "Hence, the answer is {answer}.",
]

random.seed(args.seed)

def make_assistant_flex(traces, answer):
    think = " ".join(traces)
    think = "<think>" + think + "</think>"
    conclusion = answer_bases[random.randint(0, len(answer_bases) - 1)].format(answer=answer)
    assistant = think + '\n' + conclusion
    return assistant
    
def make_assistant_fixed(traces, answer):
    think = " ".join(traces)
    think = "<think>" + think + "</think>"
    conclusion = "<answer>" + answer + "</answer>"
    assistant = think + '\n' + conclusion
    return assistant

def make_system_flex(system):
    return system
    
def make_system_fixed(system):
    format_instruction = """

## Response Format Requirements
You must provide your final answer enclosed within <answer></answer> tags:
- Example format:
  - <answer>your final answer here</answer>
- The answer tags are mandatory and must be included in your response.
"""
    return system + format_instruction


def main():
    with open(args.data_path, "r") as f:
        datas = json.load(f)
    
    flex = []
    fixed = []
    
    for data in datas:
        if args.only_format is None or args.only_format == "flex":
            flex.append({
                "system": make_system_flex(data["system"]),
                "user": data["user"],
                "assistant": make_assistant_flex(data["traces"], data["answer"]),
                "answer": data["answer"],
                "format": "flex",
            })
        if args.only_format is None or args.only_format == "fixed":
            fixed.append({
                "system": make_system_fixed(data["system"]),
                "user": data["user"],
                "assistant": make_assistant_fixed(data["traces"], data["answer"]),
                "answer": data["answer"],
                "format": "fixed",
            })
    
    data = flex + fixed
    
    import random
    random.shuffle(data)
    
    with open(args.save_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
if __name__ == "__main__":
    main()