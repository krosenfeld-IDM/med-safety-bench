#!/bin/bash

for idx in 0 3 6
do
    python generate_prompts.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --data_file_path adv_ama_prompts/adv_ama_tutorial.jsonl \
        --idx $idx \
        --output_dir datasets/Llama3/tutorial \
        --num_iter 500
done