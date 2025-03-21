#!/bin/bash

pip install -U datasets huggingface_hub evaluate

FACTOR=2
PARAMETERIZATION="teacher"
ALPHA=0.5
TEACHER_LOSS="MSE"
TEMPERATURE="2"
LAYER_SELECTION="last_k"

# google-bert/bert-base-multilingual-cased
# mbert mt - /ds/text/LangAdapters/model_fine-tune/mbert/mlt-Latn/checkpoint-99000
# mbert sw - /ds/text/LangAdapters/model_fine-tune/mbert/swh-Latn/checkpoint-96500
# mbert sk - /ds/text/LangAdapters/model_fine-tune/mbert/slk-Latn/checkpoint-95500

python distil_model.py \
  --model_name "/ds/text/LangAdapters/model_fine-tune/mbert/slk-Latn/checkpoint-95500/" \
  --tokenizer_name "google-bert/bert-base-multilingual-cased" \
  --dataset_path "data_maltese" \
  --output_dir "./models/maltese/distilled_mbert-mt_${FACTOR}_${PARAMETERIZATION}_${ALPHA}_${LAYER_SELECTION}" \
  --factor $FACTOR \
  --language_code "mlt_Latn" \
  --parameterization $PARAMETERIZATION \
  --temperature $TEMPERATURE \
  --alpha $ALPHA \
  --layer_selection $LAYER_SELECTION
