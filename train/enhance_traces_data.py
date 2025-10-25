"""
根据traces生成不同质量的数据集，用于GRPO训练

生成5种质量的数据：
1. gold: 原始完整traces
2. short: 30%随机保留trace步骤（保持顺序）
3. long: 额外增加50%的相同类型任务但来自其他样本的步骤
4. duplicate: 随机重复30%的trace，插入到相邻或最多间隔1的位置
5. bad: 随机选取其他相同task的answer替换（确保不同）

输出格式：
- system: 系统提示
- user: 用户提示
- traces: 重新组织之后的步骤
- answer: 答案
- task: 任务类型
- quality: 质量标签
"""

import os
import json
import argparse
import random
from copy import deepcopy
from collections import defaultdict

parser = argparse.ArgumentParser(description="生成不同质量的traces数据集用于GRPO训练")
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径 (all_train.json)")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)


def load_data(data_path):
    """加载训练数据"""
    print(f"加载数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 加载完成: {len(data)} 条数据")
    return data


def group_by_task(data):
    """按任务类型分组"""
    task_groups = defaultdict(list)
    for idx, item in enumerate(data):
        task_type = item.get('task', 'unknown')
        task_groups[task_type].append((idx, item))
    
    print(f"\n按任务类型分组:")
    for task_type, items in sorted(task_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  - {task_type}: {len(items)} 条")
    
    return task_groups


def generate_gold(item):
    """生成gold质量数据（原始完整traces）"""
    return {
        'system': item.get('system', ''),
        'user': item.get('user', ''),
        'traces': item.get('traces', []),
        'answer': item.get('answer', ''),
        'task': item.get('task', 'unknown'),
        'quality': 'gold'
    }


def generate_short(item, keep_ratio=0.3):
    """
    生成short质量数据（随机保留30%的trace步骤，保持顺序）
    
    Args:
        item: 原始数据项
        keep_ratio: 保留比例
    """
    traces = item.get('traces', [])
    
    if not traces:
        return generate_gold(item)
    
    # 计算保留数量（至少保留1个）
    n_keep = max(1, int(len(traces) * keep_ratio))
    
    # 随机选择要保留的索引，然后排序以保持顺序
    indices = sorted(random.sample(range(len(traces)), n_keep))
    short_traces = [traces[i] for i in indices]
    
    return {
        'system': item.get('system', ''),
        'user': item.get('user', ''),
        'traces': short_traces,
        'answer': item.get('answer', ''),
        'task': item.get('task', 'unknown'),
        'quality': 'short'
    }


def generate_long(item, task_groups, add_ratio=0.5):
    """
    生成long质量数据（额外增加50%的相同类型任务但来自其他样本的步骤）
    
    Args:
        item: 原始数据项
        task_groups: 按任务类型分组的数据
        add_ratio: 额外增加的比例
    """
    traces = item.get('traces', [])
    task_type = item.get('task', 'unknown')
    
    if not traces or task_type not in task_groups or len(task_groups[task_type]) <= 1:
        return generate_gold(item)
    
    # 计算要添加的步骤数量
    n_add = max(1, int(len(traces) * add_ratio))
    
    # 从同类型的其他样本中随机选择
    same_task_items = [other_item for _, other_item in task_groups[task_type] if other_item is not item]
    
    if not same_task_items:
        return generate_gold(item)
    
    # 随机选择其他样本
    other_item = random.choice(same_task_items)
    other_traces = other_item.get('traces', [])
    
    if not other_traces:
        return generate_gold(item)
    
    # 从其他样本中随机选择步骤
    added_traces = random.choices(other_traces, k=min(n_add, len(other_traces)))
    
    # 合并traces（插入到随机位置）
    long_traces = traces.copy()
    for trace in added_traces:
        insert_pos = random.randint(0, len(long_traces))
        long_traces.insert(insert_pos, trace)
    
    return {
        'system': item.get('system', ''),
        'user': item.get('user', ''),
        'traces': long_traces,
        'answer': item.get('answer', ''),
        'task': item.get('task', 'unknown'),
        'quality': 'long'
    }


def generate_duplicate(item, duplicate_ratio=0.3):
    """
    生成duplicate质量数据（重复30%的trace，插入到相邻或最多间隔1的位置）
    
    Args:
        item: 原始数据项
        duplicate_ratio: 重复的比例
    """
    traces = item.get('traces', [])
    
    if not traces or len(traces) < 2:
        return generate_gold(item)
    
    # 计算要重复的步骤数量
    n_duplicate = max(1, int(len(traces) * duplicate_ratio))
    
    # 随机选择要重复的步骤索引
    duplicate_indices = random.sample(range(len(traces)), min(n_duplicate, len(traces)))
    
    # 创建新的traces，插入重复步骤
    dup_traces = []
    for i, trace in enumerate(traces):
        dup_traces.append(trace)
        
        # 如果这个步骤需要重复
        if i in duplicate_indices:
            # 插入到相邻或最多间隔1的位置
            # 选项：直接插入(0)、间隔1个位置插入(1)
            gap = random.randint(0, 1)
            
            if gap == 0:
                # 直接插入重复
                dup_traces.append(deepcopy(trace))
            else:
                # 间隔1个位置后插入（如果后面还有步骤）
                if i + 1 < len(traces):
                    dup_traces.append(traces[i + 1])
                    dup_traces.append(deepcopy(trace))
    
    return {
        'system': item.get('system', ''),
        'user': item.get('user', ''),
        'traces': dup_traces,
        'answer': item.get('answer', ''),
        'task': item.get('task', 'unknown'),
        'quality': 'duplicate'
    }


def generate_bad(item, task_groups):
    """
    生成bad质量数据（随机选取其他相同task的answer替换，确保不同）
    
    Args:
        item: 原始数据项
        task_groups: 按任务类型分组的数据
    """
    task_type = item.get('task', 'unknown')
    original_answer = item.get('answer', '')
    
    if task_type not in task_groups or len(task_groups[task_type]) <= 1:
        # 如果没有其他同类型样本，返回空answer
        bad_answer = ""
    else:
        # 从同类型的其他样本中选择不同的answer
        same_task_items = [other_item for _, other_item in task_groups[task_type] if other_item is not item]
        
        # 尝试找到不同的answer
        bad_answer = original_answer
        max_attempts = 10
        for _ in range(max_attempts):
            candidate_item = random.choice(same_task_items)
            candidate_answer = candidate_item.get('answer', '')
            
            if candidate_answer != original_answer:
                bad_answer = candidate_answer
                break
    
    return {
        'system': item.get('system', ''),
        'user': item.get('user', ''),
        'traces': item.get('traces', []),
        'answer': bad_answer,
        'task': item.get('task', 'unknown'),
        'quality': 'bad'
    }


def generate_enhanced_dataset(data, task_groups, seed):
    """
    生成增强数据集（包含所有质量类型）
    
    Returns:
        list: 增强后的数据列表
    """
    set_seed(seed)
    
    enhanced_data = []
    
    print("\n生成增强数据集:")
    print("=" * 80)
    
    for idx, item in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"  处理进度: {idx + 1}/{len(data)}")
        
        # 生成5种质量的数据
        enhanced_data.append(generate_gold(item))
        enhanced_data.append(generate_short(item))
        enhanced_data.append(generate_long(item, task_groups))
        enhanced_data.append(generate_duplicate(item))
        enhanced_data.append(generate_bad(item, task_groups))
    
    print(f"✓ 生成完成: {len(data)} 条原始数据 → {len(enhanced_data)} 条增强数据")
    
    return enhanced_data


def print_statistics(enhanced_data):
    """打印统计信息"""
    quality_counts = defaultdict(int)
    task_counts = defaultdict(int)
    quality_task_counts = defaultdict(lambda: defaultdict(int))
    
    for item in enhanced_data:
        quality = item.get('quality', 'unknown')
        task = item.get('task', 'unknown')
        quality_counts[quality] += 1
        task_counts[task] += 1
        quality_task_counts[quality][task] += 1
    
    print("\n" + "=" * 80)
    print("增强数据集统计")
    print("=" * 80)
    
    print("\n按质量类型统计:")
    print("-" * 80)
    for quality, count in sorted(quality_counts.items()):
        percentage = count / len(enhanced_data) * 100
        print(f"  {quality:15s}: {count:8,} ({percentage:5.2f}%)")
    
    print(f"\n  总计: {len(enhanced_data):,} 条")
    
    print("\n按任务类型统计 (前10):")
    print("-" * 80)
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = count / len(enhanced_data) * 100
        print(f"  {task:30s}: {count:8,} ({percentage:5.2f}%)")
    
    print("\n质量-任务交叉统计 (样例):")
    print("-" * 80)
    sample_tasks = list(task_counts.keys())[:3]
    for task in sample_tasks:
        print(f"\n  任务: {task}")
        for quality in ['gold', 'short', 'long', 'duplicate', 'bad']:
            count = quality_task_counts[quality][task]
            print(f"    {quality:12s}: {count:6,} 条")
    
    print("=" * 80)


def main():
    print("=" * 80)
    print("Traces数据集质量增强工具 (用于GRPO训练)")
    print("=" * 80)
    
    # 1. 加载数据
    data = load_data(args.data_path)
    
    # 2. 按任务类型分组
    task_groups = group_by_task(data)
    
    # 3. 生成增强数据集
    enhanced_data = generate_enhanced_dataset(data, task_groups, args.seed)
    
    # 4. 打印统计信息
    print_statistics(enhanced_data)
    
    # 5. 保存结果
    print(f"\n保存增强数据集到: {args.save_path}")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 保存完成: {len(enhanced_data)} 条数据")
    
    # 6. 保存统计信息
    stats_path = args.save_path.replace('.json', '_stats.json')
    quality_counts = defaultdict(int)
    for item in enhanced_data:
        quality_counts[item.get('quality', 'unknown')] += 1
    
    stats = {
        'total': len(enhanced_data),
        'original': len(data),
        'quality_distribution': dict(quality_counts),
        'seed': args.seed
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 统计信息保存到: {stats_path}")
    
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"输入: {args.data_path} ({len(data)} 条)")
    print(f"输出: {args.save_path} ({len(enhanced_data)} 条)")
    print(f"增强倍数: {len(enhanced_data) / len(data):.1f}x")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
