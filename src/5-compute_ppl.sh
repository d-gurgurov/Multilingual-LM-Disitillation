#!/bin/bash

python get_ppl.py --models_path /netscratch/dgurgurov/projects2025/distillation_mbert/models/maltese \
                --language_code mlt_Latn \
                --output_dir pseudo_perplexity \
                --tokenizer google-bert/bert-base-multilingual-cased \
