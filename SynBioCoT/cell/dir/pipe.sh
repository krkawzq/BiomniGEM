set -e


home=/Users/wzq/Documents/Code/Project/BiomniGEM/
save_dir=${home}/data/dir/

# python ${home}/SynBioCoT/cell/de/make_qa.py \
#     --data_path "${home}/data/dir/dir.csv" \
#     --save_path "${save_dir}/qa.csv" \
#     --template_path "${home}/SynBioCoT/cell/dir/template.json" \
#     --meta_path "${home}/data/vcc_meta.json" \
#     --cell_line "H1-hESC" \
#     --clip 200

# python ${home}/SynBioCoT/make_task.py \
#     --data_path "${save_dir}/qa.csv" \
#     --save_path "${save_dir}/task.csv" \
#     --task "Cell/perturb_dir" \
#     --seed 42

# python ${home}/SynBioCoT/make_cot.py \
#     --data_path "${save_dir}/task.csv" \
#     --save_path "${save_dir}/cot.csv" \
#     --model "gpt-5-chat-latest" \
#     --max_concurrency 30

# python ${home}/SynBioCoT/make_traces.py \
#     --data_path "${save_dir}/cot.csv" \
#     --save_path "${save_dir}/traces.json" \
#     --task "expression change direction"

python ${home}/SynBioCoT/split.py \
    --data_path "${save_dir}/traces.json" \
    --save_dir "${save_dir}" \
    --split_ratio 0.8 \
    --seed 42