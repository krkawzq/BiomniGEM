#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from datasets import load_dataset

def setup_logger(log_path: str):
    logger = logging.getLogger("download_dataset")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def main():
    parser = argparse.ArgumentParser(description="下载并本地保存 HuggingFace 数据集（JSON 格式）")
    parser.add_argument("--dataset_id", type=str, default="krkawzq/SynBioCoT")
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--log_file", type=str, default="download_dataset.log")
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.dest_dir, args.log_file))

    logger.info(f"开始下载数据集 {args.dataset_id}")
    dsdict = load_dataset(args.dataset_id)  # 包含 train / validation

    # 保存为 JSON 文件
    save_root = os.path.join(args.dest_dir, "json")
    os.makedirs(save_root, exist_ok=True)
    for split in dsdict:
        out_path = os.path.join(save_root, f"{split}.json")
        logger.info(f"保存 split={split} 到 {out_path}（JSON 格式）")
        dsdict[split].to_json(out_path, force_ascii=False, orient="records", lines=True)

    logger.info("✅ 全部分割已保存（JSON 格式）。")
    logger.info(f"下次可直接用 pandas.read_json('{save_root}/train.json', lines=True) 读取。")

if __name__ == "__main__":
    main()
