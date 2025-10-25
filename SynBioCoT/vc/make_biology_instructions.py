"""
Biology-Instructions 数据清洗脚本

功能：
1. 读取原始 Biology-Instructions JSON 数据
2. 清理 instruction 中的 []标记
3. 将原始任务类型映射到合并后的任务类型
4. 提取 []标记对应的额外描述信息
5. 输出清洗后的 CSV 文件

用法：
python make_biology_instructions.py \
    --data_path /path/to/dataset_Biology_Instructions.json \
    --save_path /path/to/output/directory
"""

import pandas as pd
import json
import argparse
import os
import sys

# 导入映射配置
from bio_instructions_mapping import (
    TASK_MERGE_MAPPING,
    TASK_MODALITY,
    get_merged_task_type,
    extract_and_map_bracket_tags,
)

parser = argparse.ArgumentParser(description="清洗 Biology-Instructions 数据集")
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径 (JSON文件)")
parser.add_argument("--save_path", type=str, required=True, help="输出目录路径")
parser.add_argument("--sample_size", type=int, default=None, help="采样数量（可选，用于测试）")
args = parser.parse_args()


def clean_and_process_data(data_list, sample_size=None):
    """
    清洗和处理数据
    
    Args:
        data_list: 原始数据列表
        sample_size: 采样数量，None表示处理全部数据
        
    Returns:
        处理后的数据列表
    """
    if sample_size:
        import random
        data_list = random.sample(data_list, min(sample_size, len(data_list)))
        print(f"采样 {len(data_list)} 条数据进行处理")
    
    collected = []
    stats = {
        'total': len(data_list),
        'cleaned': 0,
        'with_bracket_tags': 0,
        'unknown_task_type': 0,
        'by_merged_task': {},
        'by_modality': {},
    }
    
    for idx, data in enumerate(data_list):
        if (idx + 1) % 100000 == 0:
            print(f"处理进度: {idx + 1}/{len(data_list)}")
        
        # 获取原始信息
        original_instruction = data["instruction"]
        original_task_type = data["metadata"]["task_type"]
        
        # 清理 instruction 并提取额外描述
        cleaned_instruction, extra_descriptions = extract_and_map_bracket_tags(original_instruction)
        
        # 获取合并后的任务类型
        merged_task_type = get_merged_task_type(original_task_type)
        
        # 获取模态信息
        modality = TASK_MODALITY.get(merged_task_type, 'unknown')
        
        # 统计信息
        stats['cleaned'] += 1
        if extra_descriptions:
            stats['with_bracket_tags'] += 1
        if merged_task_type == original_task_type:
            # 说明没有找到映射
            if merged_task_type not in TASK_MERGE_MAPPING:
                stats['unknown_task_type'] += 1
        
        stats['by_merged_task'][merged_task_type] = stats['by_merged_task'].get(merged_task_type, 0) + 1
        stats['by_modality'][modality] = stats['by_modality'].get(modality, 0) + 1
        
        # 构建输出数据
        collected.append({
            "instruction": cleaned_instruction,
            "answer": data["response"],
            "split": data["metadata"]["split"],
            "original_task_type": original_task_type,
            "task_type": merged_task_type,
            "modality": TASK_MODALITY[merged_task_type],
            "extra_descriptions": json.dumps(extra_descriptions) if extra_descriptions else "",
            "has_bracket_tags": len(extra_descriptions) > 0,
        })
    
    return collected, stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*80)
    print("数据清洗统计")
    print("="*80)
    print(f"总数据量: {stats['total']:,}")
    print(f"成功清洗: {stats['cleaned']:,}")
    print(f"包含[]标记: {stats['with_bracket_tags']:,} ({stats['with_bracket_tags']/stats['total']*100:.1f}%)")
    print(f"未知任务类型: {stats['unknown_task_type']:,}")
    
    print("\n按合并后任务类型统计:")
    print("-"*80)
    for task_type, count in sorted(stats['by_merged_task'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {task_type:35s} : {count:8,} ({count/stats['total']*100:5.2f}%)")
    
    print("\n按模态统计:")
    print("-"*80)
    for modality, count in sorted(stats['by_modality'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {modality:10s} : {count:8,} ({count/stats['total']*100:5.2f}%)")
    print("="*80 + "\n")


def main():
    # 检查输入文件
    if not os.path.exists(args.data_path):
        print(f"错误: 输入文件不存在: {args.data_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.save_path, exist_ok=True)
    print(f"输出目录: {args.save_path}")
    
    # 读取数据
    print(f"读取数据: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    print(f"原始数据量: {len(datas):,}")
    
    # 清洗和处理数据
    print("开始数据清洗...")
    collected, stats = clean_and_process_data(datas, sample_size=args.sample_size)
    
    # 打印统计信息
    print_statistics(stats)
    
    # 保存为 CSV
    df = pd.DataFrame(collected)
    output_file = os.path.join(args.save_path, "Biology_Instructions.csv")
    df.to_csv(output_file, index=False)
    print(f"✓ 数据已保存到: {output_file}")
    print(f"  - 行数: {len(df):,}")
    print(f"  - 列数: {len(df.columns)}")
    print(f"  - 列名: {', '.join(df.columns)}")
    
    # 保存统计信息
    stats_file = os.path.join(args.save_path, "cleaning_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 统计信息已保存到: {stats_file}")
    
    # 按任务类型分别保存（可选）
    print("\n按任务类型分别保存...")
    task_type_dir = os.path.join(args.save_path, "by_task_type")
    os.makedirs(task_type_dir, exist_ok=True)
    
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        task_file = os.path.join(task_type_dir, f"{task_type}.csv")
        task_df.to_csv(task_file, index=False)
        print(f"  - {task_type}: {len(task_df):,} 条")
    
    print(f"✓ 任务类型分类文件已保存到: {task_type_dir}")
    print("\n处理完成!")


if __name__ == "__main__":
    main()
