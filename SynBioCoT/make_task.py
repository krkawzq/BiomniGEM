"""
构造CoT标注任务数据

专门用于Biology-Instructions数据集
从tasks.py配置中读取文件路径、采样数量等参数
支持处理extra_descriptions列并合并到任务描述中
"""

import pandas as pd
import argparse
import os
import json
from cellm.process.template import Template
from prompt.tasks import tasks, format_list
from prompt.system import system_base, user_base

parser = argparse.ArgumentParser(description="构造CoT标注任务数据 (Biology-Instructions)")
parser.add_argument("--save_path", type=str, required=True, help="输出路径 (目录)")
parser.add_argument("--dataset", type=str, default="Biology-Instructions", help="数据集名称")
parser.add_argument("--task", type=str, default=None, help="任务名称 (可选，不指定则处理所有任务)")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
args = parser.parse_args()


def list_available_tasks(dataset_name):
    """列出指定数据集的所有可用任务"""
    if dataset_name not in tasks:
        print(f"\n❌ 错误: 数据集 '{dataset_name}' 不存在")
        print("\n可用数据集:")
        for ds_name in tasks.keys():
            print(f"  - {ds_name}")
        return None
    
    dataset_tasks = tasks[dataset_name]
    print(f"\n【{dataset_name}】可用任务:")
    print("=" * 80)
    for task_key, task_config in dataset_tasks.items():
        print(f"\n任务: {task_key}")
        print(f"  类型: {task_config.get('task_type', 'unknown')}")
        print(f"  文件: {task_config.get('file', 'N/A')}")
        print(f"  采样: {task_config.get('samples', 'all')} 条")
        print(f"  过滤: {task_config.get('filter', 'none')}")
        print(f"  摘要: {task_config['abstract'][:80]}...")
    print("=" * 80)
    return dataset_tasks


def load_data(file_path, data_root="data"):
    """
    加载数据文件
    
    Args:
        file_path: 相对于data_root的文件路径
        data_root: 数据根目录
    
    Returns:
        DataFrame
    """
    full_path = os.path.join(data_root, file_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"数据文件不存在: {full_path}")
    
    file_ext = os.path.splitext(full_path)[1].lower()
    
    if file_ext == '.csv':
        print(f"  读取CSV: {full_path}")
        return pd.read_csv(full_path)
    
    elif file_ext == '.json':
        print(f"  读取JSON: {full_path}")
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("JSON文件必须包含一个列表")
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def apply_task_filter(df, task_config):
    """应用任务特定的过滤条件"""
    if 'filter' not in task_config or not task_config['filter']:
        return df
    
    filter_expr = task_config['filter']
    print(f"  应用过滤: {filter_expr}")
    print(f"  过滤前: {len(df)} 条")
    
    try:
        df_filtered = df.query(filter_expr)
        print(f"  过滤后: {len(df_filtered)} 条")
        return df_filtered
    
    except Exception as e:
        print(f"  ⚠️ 过滤失败: {str(e)}")
        print(f"  保持原始数据")
        return df


def apply_sampling(df, task_config, seed=42):
    """应用采样（如果配置中指定了samples参数）"""
    if 'samples' not in task_config:
        return df
    
    n_samples = task_config['samples']
    
    if n_samples is None or n_samples >= len(df):
        print(f"  采样: 使用全部 {len(df)} 条")
        return df
    
    print(f"  采样: {n_samples} 条 (从 {len(df)} 条中, 种子={seed})")
    df_sampled = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    return df_sampled


def merge_extra_descriptions(df, task_description_list):
    """
    合并extra_descriptions列到任务描述中
    
    对于每一行数据：
    - 如果有extra_descriptions列且非空，则将其解析并添加到描述列表
    - 返回每行专属的完整描述
    
    Args:
        df: DataFrame
        task_description_list: 基础任务描述列表
        
    Returns:
        包含merged_descriptions列的DataFrame
    """
    if 'extra_descriptions' not in df.columns:
        # 如果没有extra_descriptions列，所有行使用相同的描述
        print("  未找到extra_descriptions列，使用基础描述")
        df['merged_descriptions'] = [task_description_list] * len(df)
        return df
    
    print(f"  检测到extra_descriptions列")
    
    merged_descriptions = []
    extra_count = 0
    
    for idx, row in df.iterrows():
        base_desc = task_description_list.copy()
        
        extra_desc_json = row['extra_descriptions']
        
        # 检查是否有额外描述
        if pd.notna(extra_desc_json) and extra_desc_json and extra_desc_json != "":
            try:
                # 解析JSON
                extra_desc = json.loads(extra_desc_json)
                
                if isinstance(extra_desc, list) and len(extra_desc) > 0:
                    # 合并到基础描述后面
                    base_desc.extend(extra_desc)
                    extra_count += 1
            
            except json.JSONDecodeError:
                # JSON解析失败，跳过
                pass
        
        merged_descriptions.append(base_desc)
    
    df['merged_descriptions'] = merged_descriptions
    print(f"  合并额外描述: {extra_count}/{len(df)} 条包含额外信息")
    
    return df


def format_task_description(description_list):
    """格式化任务描述列表"""
    if isinstance(description_list, list):
        return format_list(description_list)
    return str(description_list)


def process_single_task(dataset_name, task_key, task_config, save_dir, data_root, seed):
    """
    处理单个任务
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"处理任务: {dataset_name}/{task_key}")
    print(f"{'='*80}")
    
    try:
        # 1. 加载数据
        print("\n[1/6] 加载数据")
        df = load_data(task_config['file'], data_root)
        print(f"  ✓ 加载完成: {len(df)} 条")
        
        # 2. 应用过滤
        print("\n[2/6] 应用过滤")
        df = apply_task_filter(df, task_config)
        print(f"  ✓ 过滤完成: {len(df)} 条")
        
        # 3. 应用采样
        print("\n[3/6] 应用采样")
        df = apply_sampling(df, task_config, seed)
        print(f"  ✓ 采样完成: {len(df)} 条")
        
        # 4. 合并额外描述
        print("\n[4/6] 合并额外描述")
        df = merge_extra_descriptions(df, task_config['description'])
        print(f"  ✓ 合并完成")
        
        # 5. 应用模板
        print("\n[5/6] 应用模板")
        
        if "instruction" in df.columns:
            df["task"] = df["instruction"]
        
        template = Template({
            "system": system_base,
            "user": user_base,
            "task": "{task}",
            "answer": "{answer}",
        })
              
        # 为每一行格式化描述
        formatted_descriptions = []
        for desc_list in df['merged_descriptions']:
            formatted_descriptions.append(format_task_description(desc_list))
        
        df['task_description_formatted'] = formatted_descriptions
        
        # 应用模板，使用动态的描述
        result_rows = []
        for idx, row in df.iterrows():
            template_result = template.apply(
                pd.DataFrame([row]), 
                appends={
                    "task_abstract": task_config['abstract'],
                    "task_description_list": row['task_description_formatted']
                }
            )
            result_rows.append(template_result.iloc[0])
        
        df_final = pd.DataFrame(result_rows).reset_index(drop=True)
        print(f"  ✓ 模板应用完成")
        
        # 6. 保存结果
        print("\n[6/6] 保存结果")
        output_file = os.path.join(save_dir, f"{task_key}.csv")
        df_final.to_csv(output_file, index=False)
        print(f"  ✓ 保存到: {output_file}")
        print(f"  数据: {len(df_final)} 条 x {len(df_final.columns)} 列")
        
        return True
    
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("CoT标注任务数据构造工具 (Biology-Instructions)")
    print("="*80)
    
    # 1. 检查数据集
    dataset_name = args.dataset
    if dataset_name not in tasks:
        print(f"\n❌ 错误: 数据集 '{dataset_name}' 不存在")
        print("\n可用数据集:")
        for ds_name in tasks.keys():
            print(f"  - {ds_name}")
        exit(1)
    
    dataset_tasks = tasks[dataset_name]
    
    # 2. 确定要处理的任务
    if args.task:
        # 处理单个任务
        if args.task not in dataset_tasks:
            print(f"\n❌ 错误: 任务 '{args.task}' 不存在于数据集 '{dataset_name}'")
            list_available_tasks(dataset_name)
            exit(1)
        
        tasks_to_process = {args.task: dataset_tasks[args.task]}
        print(f"\n处理单个任务: {args.task}")
    
    else:
        # 处理所有任务
        tasks_to_process = dataset_tasks
        print(f"\n处理所有任务: {len(tasks_to_process)} 个")
        list_available_tasks(dataset_name)
    
    # 3. 创建输出目录
    os.makedirs(args.save_path, exist_ok=True)
    print(f"\n输出目录: {args.save_path}")
    
    # 4. 处理任务
    print(f"\n{'='*80}")
    print(f"开始处理")
    print(f"{'='*80}")
    
    success_count = 0
    fail_count = 0
    
    for task_key, task_config in tasks_to_process.items():
        success = process_single_task(
            dataset_name, 
            task_key, 
            task_config, 
            args.save_path, 
            args.data_root, 
            args.seed
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 5. 总结
    print(f"\n{'='*80}")
    print(f"处理完成!")
    print(f"{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"成功: {success_count}/{len(tasks_to_process)} 个任务")
    print(f"失败: {fail_count}/{len(tasks_to_process)} 个任务")
    print(f"输出目录: {args.save_path}")
    print(f"={'='*80}\n")


if __name__ == "__main__":
    main()
