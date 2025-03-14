#!/bin/bash

pip install sentencepiece

python reduce_vocab.py --dataset_path "data_maltese" \
                      --model_name "/netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbertu_2_double_loss_alpha_0.5_mse_unsoft/checkpoint-50000" \
                      --tokenizer_name "MLRS/mBERTu" \
                      --output_name "maltese_model_0.1"