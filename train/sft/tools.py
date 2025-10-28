import os
import json
import re
import random
import torch
import numpy as np
from functools import partial
from typing import List, Dict, Any
import datasets as hfds
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# basic chat_template
chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

# -- config
MAX_LEN = 4096
SEED = 42

def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def build_hf_dataset(dataset_path: str, only_gold: bool, use_assistant: bool):
    raw = _read_json_or_jsonl(dataset_path)
    rows = []
    for item in raw:
        if (not only_gold) or item.get("quality") == "gold":
            msgs = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ]
            if use_assistant:
                msgs.append({"role": "assistant", "content": item["assistant"]})
            rows.append({
                "messages": msgs,
                "answer": item.get("answer", ""),
                "quality": item.get("quality", ""),
                "format": item.get("format", ""),
            })
    return hfds.Dataset.from_list(rows)

def make_to_text(tokenizer: AutoTokenizer, add_generation_prompt: bool):
    """创建用于映射数据集的函数，使用闭包绑定 tokenizer"""
    def _f(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return {"text": text}
    return _f

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.S)
def validate_one(response: str, answer: str) -> bool:
    """验证模型输出是否匹配标准答案"""
    m = ANSWER_PATTERN.search(response or "")
    if not m:
        return False
    pred = (m.group(1) or "").strip().lower()
    gold = (answer or "").strip().lower()
    return pred == gold

def load_model(model_path: str):
    """加载模型（不手动设置设备，由 Trainer 管理）"""
    return AutoModelForCausalLM.from_pretrained(model_path)

def load_tokenizer(model_path: str, padding_side: str = "right"):
    """加载并配置 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        tokenizer.chat_template = chat_template
    else:
        try:
            _ = tokenizer.apply_chat_template([{"role": "user", "content": "test"}])
        except Exception:
            tokenizer.chat_template = chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def get_train_validate_dataset(
    train_dataset_path: str,
    validate_dataset_path: str,
    tokenizer: AutoTokenizer,
    seed: int = SEED,
    eval_keep_samples=64
):
    """准备训练集和验证集"""
    train_hf = build_hf_dataset(train_dataset_path, only_gold=True, use_assistant=True)
    eval_hf  = build_hf_dataset(validate_dataset_path, only_gold=False, use_assistant=False)

    train_hf = train_hf.shuffle(seed=seed)
    eval_hf = eval_hf.shuffle(seed=seed)
    eval_hf = eval_hf.select(range(eval_keep_samples))
    eval_answers: List[str] = eval_hf["answer"]

    train_text = train_hf.map(
        make_to_text(tokenizer, False),
        desc="Formatting train with chat_template",
        remove_columns=train_hf.column_names
    )
    eval_text = eval_hf.map(
        make_to_text(tokenizer, True),
        desc="Formatting eval with chat_template",
        remove_columns=eval_hf.column_names
    )
    return train_text, eval_text, eval_answers


def get_data_collator(tokenizer: AutoTokenizer):
    """使用 TRL 的官方 Collator，更稳健地处理标签掩码"""
    assistant_prefix = "<|im_start|>assistant\n"
    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=assistant_prefix,
    )


if __name__ == "__main__":
    VALIDATE_DATASET_PATH = "/root/autodl-fs/wzq/datasets/SynBioCoT/json/cell_validation.json"
    TRAIN_DATASET_PATH = "/root/autodl-fs/wzq/datasets/SynBioCoT/json/train.json"
    from transformers import AutoTokenizer
    MODEL_PATH = "/root/autodl-fs/wzq/models/models--SciReason--SciReasoner-8B/snapshots/772c4adaf43c750db5ef04d6f567148ca3daf7b0"
    tokenizer = load_tokenizer(MODEL_PATH)
    train, val, val_answer = get_train_validate_dataset(
        TRAIN_DATASET_PATH, VALIDATE_DATASET_PATH, tokenizer=tokenizer
    )
    print(f"训练样本数: {len(train)}，验证样本数: {len(val)}，验证答案数: {len(val_answer)}")