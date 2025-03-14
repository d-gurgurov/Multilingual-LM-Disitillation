#!/bin/bash

pip install torch
pip install adapters
pip install evaluate
pip install transformers[torch]

SCRIPT="finetune_sa.py"

mkdir -p ./tc

for SEED in 1 2 3; do
    echo "Running experiment with seed $SEED..."
    
    export PYTHONHASHSEED=$SEED
    
    python3 $SCRIPT --seed $SEED \
                    --model_path ./models/maltese/mbertu_finetuned \
                    --tokenizer_path MLRS/mBERTu \
                    --output_dir ./sa/maltese/ \
                    --dataset DGurgurov/maltese_sa

    echo "Experiment with seed $SEED completed."
done

echo "All experiments finished."
