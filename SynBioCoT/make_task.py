import pandas as pd
import argparse
import os
import json
from cellm.task import EvalTask
from cellm.process.template import Template
from prompt.tasks import tasks, format_list
from prompt.system import system_base, user_base

parser = argparse.ArgumentParser(description="构造CoT标注任务数据")
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径 (CSV或JSON)")
parser.add_argument("--save_path", type=str, required=True, help="输出路径 (CSV)")
parser.add_argument("--task", type=str, required=True, help="任务名称，格式: 'Dataset/task' 或 'task'")
parser.add_argument("--samples", type=int, default=None, help="采样数量")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
args = parser.parse_args()


def find_task_config(task_name):
    """
    根据任务名称查找任务配置
    
    Args:
        task_name: 任务名称，支持两种格式:
            - 完整格式: 'Dataset/task' (如 'Mol-Instructions/catalytic_activity')
            - 简写格式: 'task' (如 'catalytic_activity')，会自动搜索所有数据集
    
    Returns:
        tuple: (dataset_name, task_key, task_config) 或 (None, None, None)
    """
    # 尝试完整格式: Dataset/task
    if '/' in task_name:
        dataset_name, task_key = task_name.split('/', 1)
        if dataset_name in tasks and task_key in tasks[dataset_name]:
            return dataset_name, task_key, tasks[dataset_name][task_key]
        else:
            return None, None, None
    
    # 简写格式: task，搜索所有数据集
    for dataset_name, dataset_tasks in tasks.items():
        if task_name in dataset_tasks:
            return dataset_name, task_name, dataset_tasks[task_name]
    
    return None, None, None


def list_all_available_tasks():
    """列出所有可用的任务"""
    print("\n可用任务列表:")
    print("=" * 80)
    for dataset_name, dataset_tasks in tasks.items():
        print(f"\n【{dataset_name}】")
        for task_key, task_config in dataset_tasks.items():
            full_name = f"{dataset_name}/{task_key}"
            print(f"  - {full_name}")
            print(f"    类型: {task_config.get('task_type', 'unknown')}")
            print(f"    摘要: {task_config['abstract'][:60]}...")
    print("=" * 80)


def load_data(data_path):
    """
    加载数据文件
    
    支持格式:
    - CSV
    - JSON (处理后的统一格式)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.csv':
        print(f"读取CSV文件: {data_path}")
        return pd.read_csv(data_path)
    
    elif file_ext == '.json':
        print(f"读取JSON文件: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("JSON文件必须包含一个列表")
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def apply_task_filter(df, task_config):
    """
    应用任务特定的过滤条件
    
    Args:
        df: 数据DataFrame
        task_config: 任务配置字典
    
    Returns:
        过滤后的DataFrame
    
    过滤条件使用pandas query语法，例如:
        - "split == 'train'"
        - "split == 'train' and samples > 100"
        - 对于嵌套字段(如metadata.task_type)，需要先展开到列
    """
    if 'filter' not in task_config:
        return df
    
    filter_expr = task_config['filter']
    print(f"\n应用过滤条件: {filter_expr}")
    print(f"过滤前数据量: {len(df)} 条")
    
    try:
        # 直接使用pandas的query接口
        df_filtered = df.query(filter_expr)
        print(f"过滤后数据量: {len(df_filtered)} 条")
        return df_filtered
    
    except Exception as e:
        print(f"⚠️  过滤失败: {str(e)}")
        print(f"   提示: 确保filter表达式符合pandas query语法")
        print(f"   当前DataFrame列名: {list(df.columns)}")
        print(f"   保持原始数据不变")
        return df


def format_task_description(description_list):
    """格式化任务描述列表"""
    if isinstance(description_list, list):
        return "\n".join([f"{i+1}. {item}" for i, item in enumerate(description_list)])
    return str(description_list)


def main():
    print("=" * 80)
    print("CoT标注任务数据构造工具")
    print("=" * 80)
    
    # 1. 查找任务配置
    print(f"\n查找任务: {args.task}")
    dataset_name, task_key, task_config = find_task_config(args.task)
    
    if task_config is None:
        print(f"\n❌ 错误: 未找到任务 '{args.task}'")
        list_all_available_tasks()
        exit(1)
    
    full_task_name = f"{dataset_name}/{task_key}"
    print(f"✓ 找到任务: {full_task_name}")
    print(f"  类型: {task_config.get('task_type', 'unknown')}")
    print(f"  摘要: {task_config['abstract']}")
    
    # 2. 确保保存路径
    if not args.save_path.endswith(".csv"):
        print(f"\n⚠️  警告: 保存路径应该是CSV文件")
        args.save_path = args.save_path + ".csv"
    
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
    
    # 3. 加载数据
    print("\n" + "-" * 80)
    print("加载数据")
    print("-" * 80)
    df = load_data(args.data_path)
    print(f"✓ 加载完成: {len(df)} 条数据")
    print(f"  列名: {list(df.columns)}")
    
    # 4. 应用任务过滤
    df = apply_task_filter(df, task_config)
    print(f"✓ 过滤后: {len(df)} 条数据")
    
    # 5. 随机采样
    if args.samples is not None and args.samples < len(df):
        print(f"\n随机采样 {args.samples} 条 (从 {len(df)} 条中, 种子={args.seed})")
        df = df.sample(n=args.samples, random_state=args.seed)
        df = df.reset_index(drop=True)
        print(f"✓ 采样完成: {len(df)} 条")
    
    # 6. 应用模板
    print("\n" + "-" * 80)
    print("应用模板")
    print("-" * 80)
    
    # 格式化任务描述
    task_description_formatted = format_task_description(task_config['description'])
    
    template = Template({
        "system": system_base,
        "user": user_base,
        "task": "{task}",
        "answer": "{answer}"
    })
    
    df = template.apply(df, appends={
        "task_abstract": task_config['abstract'], 
        "task_description_list": task_description_formatted
    })
    
    print(f"✓ 模板应用完成")
    print(f"  新增列: system, user (如果模板中定义)")
    
    # 7. 保存结果
    print("\n" + "-" * 80)
    print("保存结果")
    print("-" * 80)
    print(f"保存到: {args.save_path}")
    df.to_csv(args.save_path, index=False)
    print(f"✓ 保存完成: {len(df)} 条数据")
    
    # 8. 显示统计信息
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"任务名称: {full_task_name}")
    print(f"输入文件: {args.data_path}")
    print(f"输出文件: {args.save_path}")
    print(f"数据条数: {len(df)}")
    print(f"数据列名: {list(df.columns)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
