#!/bin/bash

pip install -U datasets huggingface_hub evaluate torch adapters

python prune_tc_sa.py