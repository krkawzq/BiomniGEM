set -e

HOME=/Users/wzq/Documents/Code/Project/BiomniGEM/

SCRIPT=$HOME/SynBioCoT/vc

DATA=$HOME/data/vc

RAW=$HOME/VCDatasets/datasets/processed


# python $SCRIPT/make_biology_instructions.py \
#     --data_path $RAW/dataset_Biology_Instructions.json \
#     --save_path $DATA/Biology_Instructions

# python $SCRIPT/../make_task.py \
#     --save_path $DATA/Biology_Instructions/tasks \
#     --dataset "Biology-Instructions" \
#     --seed 42 \
#     --data_root $DATA/..

TASK=$DATA/Biology_Instructions/tasks

# echo "Processing enhancer_activity..."
# python $HOME/SynBioCoT/make_cot.py \
#     --data_path $TASK/enhancer_activity.csv \
#     --save_path $TASK/cot/enhancer_activity.csv \
#     --model "gpt-5-chat-latest" \
#     --max_concurrency 30

echo "Processing epigenetic_marks..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/epigenetic_marks.csv \
    --save_path $TASK/cot/epigenetic_marks.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing molecular_interaction..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/molecular_interaction.csv \
    --save_path $TASK/cot/molecular_interaction.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing promoter_detection..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/promoter_detection.csv \
    --save_path $TASK/cot/promoter_detection.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing protein_function..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/protein_function.csv \
    --save_path $TASK/cot/protein_function.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing protein_stability..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/protein_stability.csv \
    --save_path $TASK/cot/protein_stability.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing regulatory_interaction..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/regulatory_interaction.csv \
    --save_path $TASK/cot/regulatory_interaction.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing rna_function..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/rna_function.csv \
    --save_path $TASK/cot/rna_function.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing rna_isoform..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/rna_isoform.csv \
    --save_path $TASK/cot/rna_isoform.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing rna_modification..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/rna_modification.csv \
    --save_path $TASK/cot/rna_modification.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing rna_translation_regulation..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/rna_translation_regulation.csv \
    --save_path $TASK/cot/rna_translation_regulation.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Processing tf_binding..."
python $HOME/SynBioCoT/make_cot.py \
    --data_path $TASK/tf_binding.csv \
    --save_path $TASK/cot/tf_binding.csv \
    --model "gpt-5-chat-latest" \
    --max_concurrency 30

echo "Making traces for enhancer_activity..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/enhancer_activity.csv \
    --save_path $TASK/traces/enhancer_activity.json \
    --task "enhancer activity"

echo "Making traces for epigenetic_marks..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/epigenetic_marks.csv \
    --save_path $TASK/traces/epigenetic_marks.json \
    --task "epigenetic marks"

echo "Making traces for molecular_interaction..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/molecular_interaction.csv \
    --save_path $TASK/traces/molecular_interaction.json \
    --task "molecular interaction"

echo "Making traces for promoter_detection..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/promoter_detection.csv \
    --save_path $TASK/traces/promoter_detection.json \
    --task "promoter detection"

echo "Making traces for protein_function..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/protein_function.csv \
    --save_path $TASK/traces/protein_function.json \
    --task "protein function"

echo "Making traces for protein_stability..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/protein_stability.csv \
    --save_path $TASK/traces/protein_stability.json \
    --task "protein stability"

echo "Making traces for regulatory_interaction..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/regulatory_interaction.csv \
    --save_path $TASK/traces/regulatory_interaction.json \
    --task "regulatory interaction"

echo "Making traces for rna_function..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/rna_function.csv \
    --save_path $TASK/traces/rna_function.json \
    --task "rna function"

echo "Making traces for rna_isoform..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/rna_isoform.csv \
    --save_path $TASK/traces/rna_isoform.json \
    --task "rna isoform"

echo "Making traces for rna_modification..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/rna_modification.csv \
    --save_path $TASK/traces/rna_modification.json \
    --task "rna modification"

echo "Making traces for rna_translation_regulation..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/rna_translation_regulation.csv \
    --save_path $TASK/traces/rna_translation_regulation.json \
    --task "rna translation regulation"

echo "Making traces for tf_binding..."
python $HOME/SynBioCoT/make_traces.py \
    --data_path $TASK/cot/tf_binding.csv \
    --save_path $TASK/traces/tf_binding.json \
    --task "tf binding"
