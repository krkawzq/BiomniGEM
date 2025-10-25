#!/bin/zsh
set -e

home=/Users/wzq/Documents/Code/Project/BiomniGEM/
traces_dir="${home}/data/vc/Biology_Instructions/tasks/traces"
output_dir="${home}/data/vc/Biology_Instructions/tasks"

# 对 traces 目录下的所有 JSON 文件进行划分
for trace_file in ${traces_dir}/*.json; do
    filename=$(basename "$trace_file" .json)
    task_dir="${output_dir}/${filename}"
    
    # 显式创建任务目录
    mkdir -p "$task_dir"
    
    echo "Processing ${filename}..."
    python ${home}/SynBioCoT/split.py \
        --data_path "$trace_file" \
        --save_dir "$task_dir" \
        --split_ratio 0.8 \
        --seed 42
done

# 收集所有 train 数据
train_files=()
for task_dir in ${output_dir}/*/; do
    train_file="${task_dir}train.json"
    if [ -f "$train_file" ]; then
        train_files+=("$train_file")
    fi
done

python ${home}/SynBioCoT/collect_all.py \
    --data_path "${train_files[@]}" \
    --save_path "${output_dir}/train.json"

# 收集所有 val 数据
val_files=()
for task_dir in ${output_dir}/*/; do
    val_file="${task_dir}val.json"
    if [ -f "$val_file" ]; then
        val_files+=("$val_file")
    fi
done

python ${home}/SynBioCoT/collect_all.py \
    --data_path "${val_files[@]}" \
    --save_path "${output_dir}/val.json"


python ${home}/SynBioCoT/collect_all.py \
    --data_path "${home}/data/sc_ann/train.json" "${home}/data/de/train.json" "${home}/data/dir/train.json" "${output_dir}/train.json" \
    --save_path "${home}/data/all_train.json"

python ${home}/SynBioCoT/collect_all.py \
    --data_path "${home}/data/sc_ann/val.json" "${home}/data/de/val.json" "${home}/data/dir/val.json" "${output_dir}/val.json" \
    --save_path "${home}/data/all_val.json"

python ${home}/SynBioCoT/collect_all.py \
    --data_path "${home}/data/sc_ann/val.json" "${home}/data/de/val.json" "${home}/data/dir/val.json" \
    --save_path "${home}/data/cell_val.json"