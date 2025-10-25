set -e

python specialize.py \
    --model_path "/root/autodl-fs/hf-models/SciReason--SciReasoner-8B" \
    --save_path "/root/autodl-fs/hf-models/Base-SR" \
    --special_tokens "<think>" "</think>" "<cell>" "</cell>" "<dna>" "</dna>" "<rna>" "</rna>" "<protein>" "</protein>"

python specialize.py \
    --model_path "/root/autodl-fs/hf-models/Qwen--Qwen3-8B" \
    --save_path "/root/autodl-fs/hf-models/Base-Qwen" \
    --special_tokens "<think>" "</think>" "<cell>" "</cell>" "<dna>" "</dna>" "<rna>" "</rna>" "<protein>" "</protein>"