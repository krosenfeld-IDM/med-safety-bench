#!/bin/bash

python gcg.py \
    --model_path 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --data_file_path adv_ama_prompts/adv_ama_tutorial.jsonl \
    --results_file_path adv_ama_prompts/results/harmful_med_prompts.jsonl \
    --start_idx 0 \
    --num_ex 1 \
