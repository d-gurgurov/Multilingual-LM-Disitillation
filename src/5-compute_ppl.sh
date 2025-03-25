#!/bin/bash

models_group1=(
    "/netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k"
    "/netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k_12_2048"
)

models_group2=(
    "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_312"
    "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_456"
    "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_564"
)

models_group3=(
    "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_312_vocab_reduce_40k"
    "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_456_vocab_reduce_40k"
    "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_564_vocab_reduce_40k"
)

tokenizer_group1="google-bert/bert-base-multilingual-cased"

output_dir="pseudo_perplexity"
language_code="mlt_Latn"

for model in "${models_group1[@]}"; do
    python get_ppl.py --model_path "$model" \
                      --language_code "$language_code" \
                      --output_dir "$output_dir" \
                      --tokenizer "$tokenizer_group1"
done

for model in "${models_group2[@]}"; do
    python get_ppl.py --model_path "$model" \
                      --language_code "$language_code" \
                      --output_dir "$output_dir" \
                      --tokenizer "$tokenizer_group1"
done

for model in "${models_group3[@]}"; do
    python get_ppl.py --model_path "$model" \
                      --language_code "$language_code" \
                      --output_dir "$output_dir" \
                      --tokenizer "$model"
done
