#!/bin/bash

for i in {1..9}
do
    cat datasets/Llama3/prompts/dedup/category_${i}.txt datasets/Llama3/tutorial/dedup/category_${i}.txt > datasets/Llama3/combined/category_${i}.txt
done