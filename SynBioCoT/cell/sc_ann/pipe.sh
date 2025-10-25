#!/bin/zsh
set -e

home=/Users/wzq/Documents/Code/Project/BiomniGEM/
data_home=/Volumes/Wzq/Datasets/CellFM/
datasets=(
    "HumanPBMC"
    "Immune"
    "Myeloid"
    "Lung"
    "Liver"
    "Heart"
    "Skin"
)
save_dir=${home}/data/sc_ann/
cell_type_map_path=${home}/SynBioCoT/cell/sc_ann/cell_type_map.json

# 构建数据路径
data_paths=()
for dataset in "${datasets[@]}"; do
    data_paths+=("${data_home}/${dataset}.h5ad")
done
# python ${home}/SynBioCoT/cell/sc_ann/collect_data.py \
#     --data_path "${data_paths[@]}" \
#     --save_path "$save_dir/dataset.csv" \
#     --cell_type_map_path "$cell_type_map_path"

# python ${home}/SynBioCoT/cell/sc_ann/make_qa.py \
#     --data_path "${save_dir}/dataset.csv" \
#     --save_path "${save_dir}/qa.csv" \
#     --template_path "${home}/SynBioCoT/cell/sc_ann/template.json"

python ${home}/SynBioCoT/cell/sc_ann/make_task.py \
    --data_path "${save_dir}/qa.csv" \
    --save_path "${save_dir}/task.csv" \
    --task "sc_ann" \
    --samples 2000 \
    --seed 32

python ${home}/SynBioCoT/make_cot.py \
    --data_path "${save_dir}/task.csv" \
    --save_path "${save_dir}/cot.csv" \
    --model "gpt-5-chat-latest" \
    --samples 2000 \
    --max_concurrency 30

python ${home}/SynBioCoT/make_traces.py \
    --data_path "${save_dir}/cot.csv" \
    --save_path "${save_dir}/traces.json" \
    --task "single cell annotation"