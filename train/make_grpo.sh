set -e

home=/Users/wzq/Documents/Code/Project/BiomniGEM
save_dir=${home}/data/text/grpo

# Create save directory if it doesn't exist
mkdir -p "${save_dir}"

echo "Making traces data..."
python ${home}/train/enhance_traces_data.py \
    --data_path "${home}/data/all_train.json" \
    --save_path "${save_dir}/traces.json" \
    --seed 42

python ${home}/train/make_sft.py \
    --data_path "${save_dir}/traces.json" \
    --save_path "${save_dir}/train.json" \
    --only_format "flex" \
    --seed 42