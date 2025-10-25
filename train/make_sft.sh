set -e

home=/Users/wzq/Documents/Code/Project/BiomniGEM
save_dir=${home}/data/text/sft

# Create save directory if it doesn't exist
mkdir -p "${save_dir}"

echo "Making train data..."
python ${home}/train/make_sft.py \
    --data_path "${home}/data/all_train.json" \
    --save_path "${save_dir}/train.json" \
    --seed 42

echo "Making val data..."
python ${home}/train/make_sft.py \
    --data_path "${home}/data/all_val.json" \
    --save_path "${save_dir}/val.json" \
    --only_format "fixed" \
    --seed 42

echo "Making cell val data..."
python ${home}/train/make_sft.py \
    --data_path "${home}/data/cell_val.json" \
    --save_path "${save_dir}/cell_val.json" \
    --only_format "fixed" \
    --seed 42