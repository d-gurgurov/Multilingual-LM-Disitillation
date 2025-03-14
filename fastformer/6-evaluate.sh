#!/bin/bash
task_name="BoolQ"
data_dir="/netscratch/dgurgurov/projects2025/fastformer/super_glue/${task_name}"


model_type="bert"
model="/netscratch/dgurgurov/projects2025/fastformer/output_bert_prune/pruned_6_300"
out_dir="/netscratch/dgurgurov/projects2025/fastformer/output_bert_prune"


OMP_NUM_THREADS=1 python3 fastformers/examples/fastformers/run_superglue.py \
                          --model_type ${model_type} \
                          --model_name_or_path ${model} \
                          --task_name BoolQ --output_dir ${out_dir} --do_eval \
                          --data_dir ${data_dir} --per_instance_eval_batch_size 1 \
                          --do_lower_case --max_seq_length 512  \
                          --threads_per_instance 1 --no_cuda

# --use_onnxrt