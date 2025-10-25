set -e


home=/Users/wzq/Documents/Code/Project/BiomniGEM/
save_dir=${home}/data/de/

python ${home}/SynBioCoT/cell/de/make_qa.py \
    --data_path "${home}/data/de/de.csv" \
    --save_path "${save_dir}/qa.csv" \
    --template_path "${home}/SynBioCoT/cell/de/template.json" \
    --meta_path "${home}/data/vcc_meta.json" \
    --cell_line "H1-hESC" \
    --clip 200

python ${home}/SynBioCoT/make_task.py \
    --data_path "${save_dir}/qa.csv" \
    --save_path "${save_dir}/task.csv" \
    --task "Cell/perturb_de" \
    --seed 42

python ${home}/SynBioCoT/make_cot.py \
    --data_path "${save_dir}/task.csv" \
    --save_path "${save_dir}/cot.csv" \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

python ${home}/SynBioCoT/make_traces.py \
    --data_path "${save_dir}/cot.csv" \
    --save_path "${save_dir}/traces.json"