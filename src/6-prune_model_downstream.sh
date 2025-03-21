#!/bin/bash

pip install -U datasets huggingface_hub evaluate torch adapters

python prune_tc_sa.py --adapter_path "/netscratch/dgurgurov/projects2025/distillation_mbert/tc/maltese/distilled_mbert_truncate_312_tuned/1/checkpoint-880/topic_classification" \
                    --dataset_type "tc" \
                    --target_num_heads 12 \
                    --target_ffn_dim 1024 \
                    --model_path "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/tune/distilled_mbert_truncate_312_tuned/checkpoint-50000"\
                    --tokenizer_path "google-bert/bert-base-multilingual-cased" \
                    --output_dir "./pruned_models_tc"