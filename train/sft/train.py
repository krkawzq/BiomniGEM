import os, sys, math, json, time, random, argparse, importlib.util, shutil, csv, re
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup, get_scheduler,
)
from peft import LoraConfig, get_peft_model
from jinja2 import Template

chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

# =========================
# 实用工具
# =========================

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.S)

def set_seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_answer(text: str) -> str:
    if not text:
        return ""
    m = ANSWER_RE.search(text)
    return (m.group(1) if m else "").strip()

def normalize_for_compare(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

# =========================
# 训练数据：仅 assistant 反向
# =========================

def build_text_from_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """使用预定义的 chat_template 来格式化 messages"""
    try:
        # 尝试使用 tokenizer 的 chat_template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
    except Exception:
        pass
    
    # 如果没有 chat_template，使用预定义的模板
    template = Template(chat_template)
    return template.render(messages=messages, add_generation_prompt=False, tools=None)

def tokenize_with_assistant_mask(tokenizer, messages: List[Dict[str,str]], cutoff_len: int) -> Dict[str, Any]:
    """
    整体 tokenize 得到 input_ids；对每个 assistant.content 再 tokenize，
    在 input_ids 中做子序列匹配，命中的 span 置 label=input_id，其余 label=-100。
    """
    text = build_text_from_messages(tokenizer, messages)
    full = tokenizer(text, truncation=True, max_length=cutoff_len, add_special_tokens=True)
    input_ids = full["input_ids"]
    labels = [-100] * len(input_ids)

    cursor = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        a_ids = tokenizer(msg.get("content",""), add_special_tokens=False)["input_ids"]
        if not a_ids:
            continue
        end_search = len(input_ids) - len(a_ids) + 1
        for i in range(cursor, end_search):  # 修正：不能用 max(cursor, end_search)
            if input_ids[i:i+len(a_ids)] == a_ids:
                for j in range(i, i + len(a_ids)):
                    labels[j] = input_ids[j]
                cursor = i + len(a_ids)
                break

    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}

class CollatorWithLabels:
    """对 input_ids / attention_mask / labels 分别 padding；labels 的 pad 值为 -100"""
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        def pad(seq, pad_id): return seq + [pad_id] * (max_len - len(seq))
        input_ids = torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long)
        labels = torch.tensor([pad(f["labels"], -100) for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# =========================
# 验证：生成 + <answer> 解析 + 精确匹配
# =========================

def default_eval_once(
    accelerator: Accelerator,
    model, tokenizer,
    val_data: List[Dict[str,str]],
    max_new_tokens: int,
    gen_kwargs: Dict[str, Any],
    detail_writer,
) -> Tuple[int,int]:
    """
    - 用 system+user 构造 prompt (add_generation_prompt=True)
    - generate
    - 解析 <answer>...</answer>，为空判错；否则规范化精确匹配
    """
    model.eval()
    correct, total = 0, 0

    # 支持多种终止符（eos / <|im_end|>）
    eos_ids = []
    for tok_text in (tokenizer.eos_token, "<|im_end|>"):
        try:
            _id = tokenizer.convert_tokens_to_ids(tok_text)
            if isinstance(_id, int) and _id != tokenizer.unk_token_id:
                eos_ids.append(_id)
        except Exception:
            pass
    gkwargs = dict(gen_kwargs)
    if eos_ids:
        gkwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    # 左填充生成
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for ex in val_data:
            messages = []
            sys_txt = ex.get("system", "")
            if sys_txt:
                messages.append({"role": "system", "content": sys_txt})
            messages.append({"role": "user", "content": ex.get("user", "")})

            # 使用统一的 build_text_from_messages 函数，但需要 add_generation_prompt
            try:
                if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    template = Template(chat_template)
                    prompt = template.render(messages=messages, add_generation_prompt=True, tools=None)
            except Exception:
                template = Template(chat_template)
                prompt = template.render(messages=messages, add_generation_prompt=True, tools=None)
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **gkwargs)

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred_raw = extract_answer(decoded)
            gold_raw = ex.get("answer", "")

            if pred_raw == "":
                ok = False
                pred_norm = ""
                gold_norm = normalize_for_compare(gold_raw)
            else:
                pred_norm = normalize_for_compare(pred_raw)
                gold_norm = normalize_for_compare(gold_raw)
                ok = (pred_norm == gold_norm)

            if ok: correct += 1
            total += 1

            if detail_writer is not None and accelerator.is_main_process:
                detail_writer.write(json.dumps({
                    "system": sys_txt,
                    "user": ex.get("user",""),
                    "gold": gold_raw,
                    "pred_text": decoded,        # 原始生成
                    "pred_answer": pred_raw,     # <answer> 内容
                    "pred_norm": pred_norm,
                    "gold_norm": gold_norm,
                    "correct": ok
                }, ensure_ascii=False) + "\n")
    finally:
        tokenizer.padding_side = original_padding_side

    model.train()
    return correct, total

# =========================
# Checkpoint 管理
# =========================

def save_lora_and_tok(accelerator, model, tokenizer, out_dir: str):
    if accelerator.is_main_process:
        ensure_dir(out_dir)
        model.save_pretrained(out_dir)     # 仅保存 LoRA 适配器
        tokenizer.save_pretrained(out_dir)

def update_ckpt_registry(registry_file: str, step: int, acc: Optional[float], path: str) -> List[Dict[str, Any]]:
    records = []
    if os.path.exists(registry_file):
        with open(registry_file, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except Exception:
                records = []
    records.append({"step": step, "acc": acc, "path": path, "time": time.time()})
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records

def cleanup_checkpoints(accelerator: Accelerator, records: List[Dict[str, Any]], keep_best_k: int, keep_recent_n: int):
    if not accelerator.is_main_process:
        return
    # 最近 N
    recent_sorted = sorted(records, key=lambda r: r["time"], reverse=True)
    recent_keep = set(r["path"] for r in recent_sorted[:keep_recent_n])
    # 最佳 K（accuracy 优先，时间次之；相同取最新）
    acc_avail = [r for r in records if r.get("acc") is not None]
    best_sorted = sorted(acc_avail, key=lambda r: (r["acc"], r["time"]), reverse=True)
    best_keep = set(r["path"] for r in best_sorted[:keep_best_k])
    keep = recent_keep | best_keep
    for r in records:
        p = r["path"]
        if p not in keep and os.path.exists(p):
            try:
                shutil.rmtree(p)
            except Exception:
                pass

# =========================
# 主程序
# =========================

def main():
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default=None)  # 验证可选

    # 训练超参
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)  # 预留（本脚本 eval 单样本推理）
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine","linear"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens_eval", type=int, default=768)

    # 生成参数（默认即可）
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", type=str, default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    # 日志/评估/保存
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--keep_best_k", type=int, default=1)
    parser.add_argument("--keep_recent_n", type=int, default=2)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed_all(args.seed)

    # Accelerator（FSDP/精度来自 accelerate config；多卡来自 CUDA_VISIBLE_DEVICES）
    accelerator = Accelerator()
    main_process = accelerator.is_main_process
    if main_process:
        print(f"[INFO] world_size={accelerator.num_processes}, device={accelerator.device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 训练数据（ShareGPT）
    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    def map_train(ex):
        messages = ex.get("messages", None)
        if not isinstance(messages, list):
            raise ValueError("训练数据必须是 ShareGPT 格式：每行包含 messages 列表")
        return tokenize_with_assistant_mask(tokenizer, messages, args.cutoff_len)

    train_tok = train_ds.map(map_train, remove_columns=train_ds.column_names, num_proc=32)
    collator = CollatorWithLabels(tokenizer)
    train_loader = DataLoader(
        train_tok,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    # 验证数据（system/user/answer）
    val_data = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_raw = load_dataset("json", data_files=args.val_jsonl, split="train")
        val_data = [{"system":ex.get("system",""), "user":ex.get("user",""), "answer":ex.get("answer","")} for ex in val_raw]

    # 模型 + LoRA
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[s.strip() for s in args.lora_target.split(",") if s.strip()],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    # 优化器 & 调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    max_steps = updates_per_epoch * args.num_train_epochs
    warmup_steps = int(max_steps * args.warmup_ratio)
    if args.lr_scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    else:
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    # 分布式准备
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    # 日志与登记
    metrics_dir = os.path.join(args.output_dir, "metrics")
    ensure_dir(metrics_dir)
    train_log = os.path.join(metrics_dir, "train.log")
    train_csv = os.path.join(metrics_dir, "training_metrics.csv")
    eval_csv = os.path.join(metrics_dir, "eval_summary.csv")
    eval_detail = os.path.join(metrics_dir, "eval_detail.jsonl")
    ckpt_registry = os.path.join(args.output_dir, "checkpoint_registry.json")

    if main_process and not os.path.exists(train_csv):
        with open(train_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["global_step","epoch","loss","lr","time"])

    gen_kwargs = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    # 训练
    model.train()
    global_step, best_acc_seen = 0, None

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 日志
            if main_process and (step % args.logging_steps == 0 or step == 1):
                with open(train_log, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | epoch {epoch} step {step} loss {loss.item():.6f} lr {lr_scheduler.get_last_lr()[0]:.2e}\n")
                with open(train_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([global_step, epoch, f"{loss.item():.6f}", f"{lr_scheduler.get_last_lr()[0]:.2e}", time.strftime("%Y-%m-%d %H:%M:%S")])

            # 仅在同步梯度的 step 递增一次
            if accelerator.sync_gradients:
                global_step += 1

                # 保存 checkpoint（LoRA + tokenizer）
                if (global_step % args.save_steps == 0) and main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_lora_and_tok(accelerator, model, tokenizer, ckpt_dir)
                    records = update_ckpt_registry(ckpt_registry, global_step, None, ckpt_dir)
                    # 先按“最近 N”做一次清理（评估后再做最佳 K）
                    cleanup_checkpoints(accelerator, records, keep_best_k=0, keep_recent_n=args.keep_recent_n)

                # 评估（仅主进程）
                if (val_data is not None) and (global_step % args.eval_steps == 0) and main_process:
                    with open(eval_detail, "a", encoding="utf-8") as detail_writer:
                        correct, total = default_eval_once(
                            accelerator, model, tokenizer,
                            val_data, max_new_tokens=args.max_new_tokens_eval,
                            gen_kwargs=gen_kwargs, detail_writer=detail_writer
                        )
                    acc = (correct / max(1, total)) if total else 0.0
                    if not os.path.exists(eval_csv):
                        with open(eval_csv, "w", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow(["global_step","correct","total","accuracy","time"])
                    with open(eval_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([global_step, correct, total, f"{acc:.6f}", time.strftime("%Y-%m-%d %H:%M:%S")])

                    # 更新登记并清理（最佳 K + 最近 N）
                    records = []
                    if os.path.exists(ckpt_registry):
                        with open(ckpt_registry, "r", encoding="utf-8") as rf:
                            try:
                                records = json.load(rf)
                            except Exception:
                                records = []
                    if records and records[-1]["step"] == global_step and records[-1].get("acc") is None:
                        records[-1]["acc"] = acc
                        with open(ckpt_registry, "w", encoding="utf-8") as wf:
                            json.dump(records, wf, ensure_ascii=False, indent=2)
                    cleanup_checkpoints(accelerator, records, keep_best_k=args.keep_best_k, keep_recent_n=args.keep_recent_n)

                    if best_acc_seen is None or acc >= best_acc_seen:
                        best_acc_seen = acc
                        print(f"[Eval] step={global_step} acc={acc:.4f} (best)")

    # 最终保存
    if main_process:
        final_dir = os.path.join(args.output_dir, "final")
        save_lora_and_tok(accelerator, model, tokenizer, final_dir)
        print(f"[DONE] final adapter saved to: {final_dir}")

if __name__ == "__main__":
    main()
