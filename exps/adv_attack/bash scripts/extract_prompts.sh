#!/bin/bash

for subdir in "prompts" "tutorial"
do
    data_dir="datasets/Llama3/${subdir}"

    for i in {1..9}
    do
        python extract_prompts.py "${data_dir}/responses_cat_${i}.txt" "${data_dir}/category_${i}.txt"
    done

    python dataset_cleanup.py $data_dir
done