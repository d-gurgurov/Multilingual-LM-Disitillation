#!/bin/bash

pip install -U datasets huggingface_hub evaluate

# Define variables
FACTOR=2
PARAMETERIZATION="teacher"

# Run Python script with variables
python train_model_small.py \
  --model_name "MLRS/mBERTu" \
  --tokenizer_name "MLRS/mBERTu" \
  --dataset_path "data_maltese" \
  --output_dir "./models/maltese/student_mbertu_${FACTOR}_${PARAMETERIZATION}" \
  --factor $FACTOR \
  --parameterization $PARAMETERIZATION
