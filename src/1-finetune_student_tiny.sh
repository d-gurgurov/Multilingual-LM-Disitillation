#!/bin/bash

pip install -U datasets huggingface_hub evaluate

FACTOR=2
PARAMETERIZATION="teacher"
LAYER_SELECTION="last_k"
TINY_MODEL_INIT="svd" # svd
INTERMEDIATE=312 # 128, 256, 312

# google-bert/bert-base-multilingual-cased
# mbert mt - /ds/text/LangAdapters/model_fine-tune/mbert/mlt-Latn/checkpoint-99000
# mbert sw - /ds/text/LangAdapters/model_fine-tune/mbert/swh-Latn/checkpoint-96500
# mbert sk - /ds/text/LangAdapters/model_fine-tune/mbert/slk-Latn/checkpoint-95500

python train_model_small.py \
  --model_name "/ds/text/LangAdapters/model_fine-tune/mbert/mlt-Latn/" \
  --tokenizer_name "google-bert/bert-base-multilingual-cased" \
  --dataset_path "data_maltese" \
  --output_dir "./tiny_models/maltese/student_tiny_mbert-mt_${TINY_MODEL_INIT}_${INTERMEDIATE}_big_teacher" \
  --factor $FACTOR \
  --language_code "mlt_Latn" \
  --parameterization $PARAMETERIZATION \
  --layer_selection $LAYER_SELECTION \
  --student_model_path "/netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k_12_2048/checkpoint-100" \
  --tiny_model_init $TINY_MODEL_INIT \
  --hidden_size $INTERMEDIATE
