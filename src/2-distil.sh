#!/bin/bash

pip install -U datasets huggingface_hub evaluate

# Define variables
FACTOR=2
PARAMETERIZATION="teacher"
ALPHA=0.5
TEACHER_LOSS="KL"
TEMPERATURE="2"

# Run Python script with variables
python distil_mbert_double_loss.py \
  --model_name "MLRS/mBERTu" \
  --tokenizer_name "MLRS/mBERTu" \
  --dataset_path "data_maltese" \
  --output_dir "./models/maltese/distilled_mbertu_${FACTOR}_${PARAMETERIZATION}_${ALPHA}" \
  --factor $FACTOR \
  --parameterization $PARAMETERIZATION \
  --temperature $TEMPERATURE \
  --alpha $TEACHER_LOSS
