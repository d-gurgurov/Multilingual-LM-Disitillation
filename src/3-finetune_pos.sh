#!/bin/bash

pip install torch
pip install adapters
pip install transformers==4.46.0
pip install accelerate -U
pip install datasets==v2.21.0
pip install seqeval
pip install conllu

# Single language configuration
language='mlt_Latn'

# List of models to evaluate
models=(
  "bert-base-multilingual-cased"
  "DGurgurov/mbert_mlt-latn"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k_12_2048"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_312"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_456"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_564"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_312_vocab_reduce_40k"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_456_vocab_reduce_40k"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_564_vocab_reduce_40k"
)

# Model names (should match the order of models array)
model_names=(
  "bert-base-multilingual-cased"
  "mbert_mlt-latn"
  "distilled_mbert-mt_2_teacher_0.5_last_k"
  "distilled_mbert-mt_2_teacher_0.5_last_k_12_2048"
  "distilled_tiny_mbert-mt_truncation_312"
  "distilled_tiny_mbert-mt_truncation_456"
  "distilled_tiny_mbert-mt_truncation_564"
  "distilled_tiny_mbert-mt_truncation_312_vocab_reduce_40k"
  "distilled_tiny_mbert-mt_truncation_456_vocab_reduce_40k"
  "distilled_tiny_mbert-mt_truncation_564_vocab_reduce_40k"
)

# Common tokenizer for all models
tokenizers=(
  "bert-base-multilingual-cased"
  "bert-base-multilingual-cased"
  "bert-base-multilingual-cased"
  "bert-base-multilingual-cased"
  "bert-base-multilingual-cased"
  "bert-base-multilingual-cased"
  "bert-base-multilingual-cased"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_312_vocab_reduce_40k"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_456_vocab_reduce_40k"
  "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_564_vocab_reduce_40k"
)

# Directory base path
base_dir="/netscratch/dgurgurov/projects2025/distillation_mbert"

# Loop over each model and seed
for i in "${!models[@]}"; do
  model="${models[$i]}"
  model_name="${model_names[$i]}"
  tokenizer="${tokenizers[$i]}"
  
  for seed in 1 2 3; do
    # Define output directory
    output_dir="${base_dir}/pos/${language}/${model_name}/${seed}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    echo "Running model: ${model_name}, seed: ${seed}"

    # Run the script with the specified arguments
    python finetune_pos.py \
      --language "$language" \
      --output_dir "$output_dir" \
      --model_name "$model" \
      --tokenizer_name "$tokenizer" \
      --learning_rate 3e-4 \
      --num_train_epochs 100 \
      --per_device_train_batch_size 64 \
      --per_device_eval_batch_size 64 \
      --evaluation_strategy "epoch" \
      --save_strategy "epoch" \
      --weight_decay 0.01 \
      --seed "$seed" \
      --ud_train_file "pos/mlt_Latn/mt_mudt-ud-train.conllu" \
      --ud_dev_file "pos/mlt_Latn/mt_mudt-ud-dev.conllu" \
      --ud_test_file "pos/mlt_Latn/mt_mudt-ud-test.conllu"
  done
done