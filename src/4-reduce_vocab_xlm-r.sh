#!/bin/bash

# Install dependencies
pip install sentencepiece torch datasets
pip install transformers==4.37
pip install lm-vocab-trimmer/

# copy and paste in the original tokenizer to the model folder if not there

# Define models and their configurations
declare -A models=(
  ["maltese"]="/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_xlm-r-mt_truncation_312/checkpoint-46500 /netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_xlm-r-mt_truncation_456/checkpoint-43500 /netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese/distilled_tiny_xlm-r-mt_truncation_564/checkpoint-43500"
  ["slovak"]="/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/slovak/distilled_tiny_xlm-r-sk_truncation_312/checkpoint-125000 /netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/slovak/distilled_tiny_xlm-r-sk_truncation_456/checkpoint-125000 /netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/slovak/distilled_tiny_xlm-r-sk_truncation_564/checkpoint-125000"
  ["swahili"]="/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/swahili/distilled_tiny_xlm-r-sw_truncation_312/checkpoint-83000 /netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/swahili/distilled_tiny_xlm-r-sw_truncation_456/checkpoint-96000 /netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/swahili/distilled_tiny_xlm-r-sw_truncation_564/checkpoint-101000"
)

declare -A languages=(
  ["maltese"]="mt"
  ["slovak"]="sk"
  ["swahili"]="sw"
)

declare -A data_folders=(
  ["maltese"]="data_maltese"
  ["slovak"]="data_slovak"
  ["swahili"]="data_swahili"
)

# Loop through each language and corresponding models
for language in "${!models[@]}"; do
  models_list=(${models[$language]})
  lang_code="${languages[$language]}"
  data_folder="${data_folders[$language]}"
  
  for model in "${models_list[@]}"; do
    # Extract truncation value from model path
    if [[ $model =~ truncation_([0-9]+) ]]; then
      trunc="${BASH_REMATCH[1]}"
    else
      echo "Warning: Could not extract truncation size from $model"
      continue
    fi

    # Construct save path
    save_path="reduce_vocab/distilled_tiny_xlm-r-${lang_code}_truncation_${trunc}_vocab_reduce_40k"
    
    # Run trimming
    vocabtrimmer-trimming -m "$model" -d "$data_folder" -s train -v 40000 -p "$save_path" --dataset-column content -l "$lang_code"
  done
done