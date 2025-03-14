#!/bin/bash
task="BoolQ"
data_dir="/netscratch/dgurgurov/projects2025/fastformer/super_glue/${task}"
out_dir="/netscratch/dgurgurov/projects2025/fastformer/output_bert_prune"

model_type="bert"
model="/netscratch/dgurgurov/projects2025/fastformer/models/teacher-bert-base"

gpuid=0
seed=42
seq_len_train=512
state_loss_ratio=0.1


python fastformers/examples/fastformers/run_superglue.py \
        --data_dir ${data_dir} --task_name ${task} \
        --output_dir ${out_dir} --model_type ${model_type} \
        --model_name_or_path ${model} --do_eval \
        --do_prune --max_seq_length ${seq_len_train} \
        --per_instance_eval_batch_size 1 \
        --target_num_heads 6 --target_ffn_dim 300