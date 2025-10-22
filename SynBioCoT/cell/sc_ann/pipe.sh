#!/bin/zsh
set -e

data_home="/Volumes/Wzq/Datasets/CellFM"
datasets=(
    "HumanPBMC"
    "Immune"
    "Myeloid"
    "Lung"
    "Liver"
    "Heart"
    "Skin"
)
save_path="./dataset.csv"
cell_type_map_path="./cell_type_map.json"

# 构建数据路径
data_paths=()
for dataset in "${datasets[@]}"; do
    data_paths+=("${data_home}/${dataset}.h5ad")
done
python SynBioCoT/cell/sc_ann/collect_data.py \
    --data_path "${data_paths[@]}" \
    --save_path "$save_path" \
    --cell_type_map_path "$cell_type_map_path"
