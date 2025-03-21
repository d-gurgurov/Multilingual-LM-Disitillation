#!/bin/bash

pip install -U datasets huggingface_hub evaluate torch adapters

# mt - /netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese/distilled_mbert-mt_2_teacher_0.5_last_k/checkpoint-48500
# sl - /netscratch/dgurgurov/projects2025/distillation_mbert/models/slovak/distilled_mbert-sk_2_teacher_0.5_last_k/checkpoint-122500
# sw - /netscratch/dgurgurov/projects2025/distillation_mbert/models/swahili/distilled_mbert-sw_2_teacher_0.5_last_k/checkpoint-92500

python prune_mlm.py \
                    --target_num_heads 12 \
                    --target_ffn_dim 2048 \
                    --model_path "/netscratch/dgurgurov/projects2025/distillation_mbert/models/swahili/distilled_mbert-sw_2_teacher_0.5_last_k/checkpoint-92500"\
                    --tokenizer_path "google-bert/bert-base-multilingual-cased" \
                    --output_dir "/netscratch/dgurgurov/projects2025/distillation_mbert/models" \
                    --language "swh_Latn"