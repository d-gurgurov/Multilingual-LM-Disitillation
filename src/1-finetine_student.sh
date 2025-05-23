#!/bin/bash

pip install -U datasets huggingface_hub evaluate

FACTOR=2
PARAMETERIZATION="teacher"
LAYER_SELECTION="last_k"

# google-bert/bert-base-multilingual-cased
# mbert mt - /ds/text/LangAdapters/model_fine-tune/mbert/mlt-Latn/checkpoint-99000
# mbert sw - /ds/text/LangAdapters/model_fine-tune/mbert/swh-Latn/checkpoint-96500
# mbert sk - /ds/text/LangAdapters/model_fine-tune/mbert/slk-Latn/checkpoint-95500

python train_model_small.py \
  --model_name "/ds/text/LangAdapters/model_fine-tune/mbert/mlt-Latn/checkpoint-99000" \
  --tokenizer_name "google-bert/bert-base-multilingual-cased" \
  --dataset_path "data_maltese" \
  --output_dir "./tiny_models/maltese/student_mbert-mt_${FACTOR}_${PARAMETERIZATION}_${LAYER_SELECTION}" \
  --factor $FACTOR \
  --language_code "mlt_Latn" \
  --parameterization $PARAMETERIZATION \
  --layer_selection $LAYER_SELECTION 
