import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForMaskedLM, BertConfig, AutoModel

def count_embedding_params(model):
    """
    Count the number of parameters in the embedding layer and total.
    
    Args:
        model: BERT model
        
    Returns:
        (embedding_params, total_params): Tuple of parameter counts
    """
    embedding_params = sum(p.numel() for name, p in model.named_parameters() 
                          if 'embeddings' in name)
    total_params = sum(p.numel() for p in model.parameters())
    return embedding_params, total_params
        
def modify_bert_model(model, hidden_size=312, intermediate_size=2048, num_attention_heads=12):
    config = model.config
    
    # Update model config
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_attention_heads = num_attention_heads  
    
    # Ensure hidden_size is divisible by num_attention_heads
    assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads!"
    
    # Reinitialize model with modified config
    new_model = AutoModelForMaskedLM.from_config(config)
    
    return new_model

# Load the original model
model_path = "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/distilled_tiny_mbert-mt_truncation_564_vocab_reduce/checkpoint-100"
original_model = AutoModelForMaskedLM.from_pretrained(model_path)

# Modify the model dimensions
modified_model = modify_bert_model(original_model, hidden_size=312, intermediate_size=3072, num_attention_heads=12)

# Get parameter counts
embedding_params, total_params = count_embedding_params(original_model)

print(f"Total parameters: {total_params / 1e6:.2f}M")
print(f"Embedding parameters: {embedding_params / 1e6:.2f}M")
print(f"Percentage of total: {embedding_params / total_params * 100:.2f}%")

print(original_model)
