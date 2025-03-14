#!/bin/bash

pip install -U datasets huggingface_hub evaluate torch adapters

python prune_tc_sa.py --adapter_path "/netscratch/dgurgurov/projects2025/distillation_mbert/sa/distilled_mbertu_2_double_loss_alpha_0.5_mse_unsoft/1/checkpoint-760/sentiment_classification" \
                    --dataset_type "sa" \
                    --target_num_heads 12 \
                    --target_ffn_dim 2048 \
                    --model_path "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/maltese_model_0.1"\
                    --tokenizer_path "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/maltese_model_0.1" \
                    --output_dir "./pruned_models_sa"