#!/bin/bash

pip install -U datasets huggingface_hub evaluate

python train_model_full.py \
  --model_name "MLRS/mBERTu" \
  --tokenizer_name "MLRS/mBERTu" \
  --dataset_path "data_maltese" \
  --output_dir "/models/maltese/mbertu_finetuned" \
  --language_code "mlt_Latn"
