#!/bin/bash

python get_ppl.py --models_path models/maltese \
                --tokenizer google-bert/bert-base-multilingual-cased \
                --language_code mlt_Latn \
                --output_dir pseudo_perplexity
