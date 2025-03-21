#!/bin/bash

pip install sentencepiece

# mt - /netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k_12_2048
# sk - /netscratch/dgurgurov/projects2025/distillation_mbert/models/slovak/distilled_mbert-sk_2_teacher_0.5_last_k_12_2048
# sw - /netscratch/dgurgurov/projects2025/distillation_mbert/models/swahili/distilled_mbert-sw_2_teacher_0.5_last_k_12_2048

python reduce_vocab.py --dataset_path "data_maltese" \
                      --model_name "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_mbert-mt_truncation_312/checkpoint-50000" \
                      --tokenizer_name "google-bert/bert-base-multilingual-cased" \
                      --output_name "distilled_tiny_mbert-mt_truncation_312_vocab_reduce" \
                      --target_vocab_size 40000