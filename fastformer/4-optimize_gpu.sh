#!/bin/bash
task="BoolQ"

model_type="bert"
model="/netscratch/dgurgurov/projects2025/fastformer/models/teacher-bert-base"


python3 fastformers/examples/fastformers/run_superglue.py \
        --task_name ${task} \
        --model_type ${model_type} \
        --model_name_or_path ${model} \
        --convert_fp16