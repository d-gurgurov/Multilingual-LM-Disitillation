#!/bin/bash

pip install -U datasets huggingface_hub evaluate

ALPHA=0.5
TINY_MODEL_INIT="truncation" # svd
INTERMEDIATE=312 # 312, 456, 564

python distil_model.py \
  --model_name "/netscratch/dgurgurov/projects2025/distillation_mbert/models/slovak/distilled_mbert-sk_2_teacher_0.5_last_k_12_2048" \
  --tokenizer_name "google-bert/bert-base-multilingual-cased" \
  --dataset_path "data_slovak" \
  --output_dir "./tiny_models/slovak/distilled_tiny_mbert-sk_${TINY_MODEL_INIT}_${INTERMEDIATE}" \
  --language_code "slk_Latn" \
  --factor 2 \
  --parameterization "teacher" \
  --alpha $ALPHA \
  --student_model_path "/netscratch/dgurgurov/projects2025/distillation_mbert/models/slovak/distilled_mbert-sk_2_teacher_0.5_last_k_12_2048" \
  --tiny_model_init $TINY_MODEL_INIT \
  --hidden_size $INTERMEDIATE
