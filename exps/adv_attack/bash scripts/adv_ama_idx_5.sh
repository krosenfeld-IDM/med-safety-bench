#!/bin/bash

python gcg.py \
    --model_path 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --data_file_path adv_ama_prompts/adv_ama_tutorial.jsonl \
    --results_file_path adv_ama_prompts/results/harmful_med_prompts.jsonl \
    --start_idx 5 \
    --num_ex 1 \

# python gcg.py \
#     --data_file_path adv_ama_prompts/cat6.jsonl \
#     --results_file_path adv_ama_prompts/results/harmful_med_prompts.jsonl \
#     --commit_hash "c1b0db933684edbfe29a06fa47eb19cc48025e93" \
#     --start_idx 5 \
#     --num_ex 1 \
