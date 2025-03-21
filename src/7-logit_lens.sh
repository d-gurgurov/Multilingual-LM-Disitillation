#!/bin/bash

pip install --upgrade pip
pip install torch transformers numpy matplotlib seaborn adapters

# Models to try out: 
#   google-bert/bert-base-multilingual-cased 
#   MLRS/mBERTu 
#   FacebookAI/xlm-roberta-base MLRS/mBERTu

python logit_lens.py --model_name "google-bert/bert-base-multilingual-cased" \
                     --text "Много добър. Заслужава си определено. Определено заслужаваше Оскар." \
                     --visualization "all"