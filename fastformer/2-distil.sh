#!/bin/bash

task_name="BoolQ"
data_dir="/netscratch/dgurgurov/projects2025/fastformer/super_glue/${task_name}"
out_dir="/netscratch/dgurgurov/projects2025/fastformer/output_bert_distil"

teacher_model_type="bert"
teacher_model="/netscratch/dgurgurov/projects2025/fastformer/models/teacher-bert-base"

student_model_type="bert"
student_model="distilbert/distilbert-base-uncased"

gpuid=0
seed=42
seq_len_train=512
eval_freq=500
state_loss_ratio=0.1

# Run distillation
python fastformers/examples/fastformers/run_superglue.py \
        --data_dir ${data_dir} \
        --task_name ${task_name} \
        --output_dir ${out_dir} \
        --teacher_model_type ${teacher_model_type} \
        --teacher_model_name_or_path ${teacher_model} \
        --model_type ${student_model_type} \
        --model_name_or_path ${student_model} \
        --use_gpuid ${gpuid} \
        --seed ${seed} \
        --do_train \
        --max_seq_length ${seq_len_train} \
        --do_eval \
        --eval_and_save_steps ${eval_freq} \
        --save_only_best \
        --learning_rate 0.00001 \
        --warmup_ratio 0.06 \
        --weight_decay 0.01 \
        --per_gpu_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --logging_steps 100 \
        --num_train_epochs 10 \
        --overwrite_output_dir \
        --per_instance_eval_batch_size 8 \
        --state_loss_ratio ${state_loss_ratio}
