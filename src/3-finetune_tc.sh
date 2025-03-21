#!/bin/bash

pip install torch
pip install adapters
pip install evaluate
pip install transformers[torch]

SCRIPT="finetune_tc.py"

mkdir -p ./tc

for SEED in 1 2 3; do
    echo "Running experiment with seed $SEED..."

    export PYTHONHASHSEED=$SEED

    python3 $SCRIPT --seed $SEED \
                    --model_path google-bert/bert-base-multilingual-cased \
                    --output_dir ./tc/maltese/ \
                    --language mlt_Latn \
                    --tokenizer_path google-bert/bert-base-multilingual-cased
                    # --tokenizer_path /netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_mbert-mt_2_teacher_0.5_last_k_12_2048_vocab_reduce/checkpoint-100 \

    echo "Experiment with seed $SEED completed."
done

echo "All experiments finished."
