#!/bin/bash

# download superglue data from the original website

pip install onnxruntime 
pip uninstall transformers -y
git clone https://github.com/microsoft/fastformers
cd fastformers
pip install .

pip install numpy==1.23.4
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113