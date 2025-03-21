import os
import torch
import numpy as np
import random
from tqdm import tqdm
from heapq import heappush, heappop
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, SequentialSampler
import logging
from datasets import load_dataset
import argparse

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sort_by_importance(weight, bias, importance, num_instances, stride):
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)), )
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)), )
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat(sorted_bias_to_concat) if sorted_bias_to_concat is not None else None

def compute_mlm_metrics(eval_loss, eval_masked_lm_accuracy, eval_steps):
    """Compute MLM metrics."""
    perplexity = torch.exp(torch.tensor(eval_loss / eval_steps))
    return {
        "perplexity": perplexity.item(),
        "loss": eval_loss / eval_steps,
        "masked_lm_accuracy": eval_masked_lm_accuracy / eval_steps
    }

def evaluate_mlm_model(model, dataset, tokenizer, device, batch_size=8):
    """Evaluate MLM model on a dataset."""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, 
        sampler=eval_sampler, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    model.eval()
    eval_loss = 0.0
    eval_masked_lm_accuracy = 0.0
    eval_steps = 0
    
    for batch in tqdm(eval_dataloader, desc="Evaluating MLM"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            
        eval_loss += outputs.loss.item()
        
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        
        mask = labels != -100
        masked_preds = predictions[mask]
        masked_labels = labels[mask]
        
        if len(masked_preds) > 0:
            accuracy = (masked_preds == masked_labels).float().mean().item()
            eval_masked_lm_accuracy += accuracy
        
        eval_steps += 1
    
    metrics = compute_mlm_metrics(eval_loss, eval_masked_lm_accuracy, eval_steps)
    
    return metrics

def prune_model(model_path, val_dataset, tokenizer, target_num_heads, target_ffn_dim, output_dir, device="cuda"):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Evaluating MLM model before pruning...")
    before_metrics = evaluate_mlm_model(model, val_dataset, tokenizer, device)
    print(f"Before pruning: {before_metrics}")
    
    head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)
    
    head_importance = torch.zeros(config.num_hidden_layers, config.num_attention_heads).to(device)
    ffn_importance = torch.zeros(config.num_hidden_layers, config.intermediate_size).to(device)
    
    layers = model.bert.encoder.layer if hasattr(model, 'bert') else model.roberta.encoder.layer
    
    inter_weights = torch.zeros(config.num_hidden_layers, config.intermediate_size, config.hidden_size).to(device)
    inter_biases = torch.zeros(config.num_hidden_layers, config.intermediate_size).to(device)
    
    for layer_num in range(config.num_hidden_layers):
        inter_weights[layer_num] = layers[layer_num].intermediate.dense.weight.detach().to(device)
        inter_biases[layer_num] = layers[layer_num].intermediate.dense.bias.detach().to(device)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    eval_sampler = SequentialSampler(val_dataset)
    eval_dataloader = DataLoader(
        val_dataset, 
        sampler=eval_sampler, 
        batch_size=8, 
        collate_fn=data_collator
    )
    
    model.eval()
    tot_tokens = 0.0
    
    print("Computing importance scores for pruning...")
    for batch in tqdm(eval_dataloader, desc="Computing importance"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        model.zero_grad()
        
        # Forward pass with head mask to get attention outputs
        # We need to adapt this to use MLM loss instead of classification loss
        model_kwargs = {
            "head_mask": head_mask,
            "output_attentions": True
        }
        outputs = model(**batch, **model_kwargs)
        loss = outputs.loss
        
        # Backward pass to get gradients
        loss.backward()
        
        # Accumulate head importance scores
        head_importance += head_mask.grad.abs().detach()
        
        # Accumulate FFN importance scores
        for layer_num in range(config.num_hidden_layers):
            ffn_importance[layer_num] += torch.abs(
                torch.sum(layers[layer_num].intermediate.dense.weight.grad.detach() * inter_weights[layer_num], 1) 
                + layers[layer_num].intermediate.dense.bias.grad.detach() * inter_biases[layer_num]
            )
        
        tot_tokens += batch["attention_mask"].float().detach().sum().data
    
    # Normalize importance scores
    head_importance /= tot_tokens
    
    # Layerwise importance normalization
    exponent = 2
    norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
    head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
    
    # Move importance scores to CPU for rewiring
    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    
    # Calculate head size
    head_size = config.hidden_size // config.num_attention_heads
    
    print("Rewiring the network...")
    # Rewire the network
    for layer_num in range(config.num_hidden_layers):
        # Get query, key, value weights and biases
        attention_self = layers[layer_num].attention.self
        query_weight = attention_self.query.weight
        query_bias = attention_self.query.bias
        key_weight = attention_self.key.weight
        key_bias = attention_self.key.bias
        value_weight = attention_self.value.weight
        value_bias = attention_self.value.bias
        
        # Sort query, key, value based on importance scores
        query_weight, query_bias = sort_by_importance(
            query_weight, 
            query_bias,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        attention_self.query.weight = torch.nn.Parameter(query_weight)
        attention_self.query.bias = torch.nn.Parameter(query_bias)
        
        key_weight, key_bias = sort_by_importance(
            key_weight,
            key_bias,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        attention_self.key.weight = torch.nn.Parameter(key_weight)
        attention_self.key.bias = torch.nn.Parameter(key_bias)
        
        value_weight, value_bias = sort_by_importance(
            value_weight,
            value_bias,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        attention_self.value.weight = torch.nn.Parameter(value_weight)
        attention_self.value.bias = torch.nn.Parameter(value_bias)
        
        # Sort attention output matrix
        weight_sorted, _ = sort_by_importance(
            layers[layer_num].attention.output.dense.weight.transpose(0, 1),
            None,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        weight_sorted = weight_sorted.transpose(0, 1)
        layers[layer_num].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)
        
        # Sort intermediate (FFN) layer
        weight_sorted, bias_sorted = sort_by_importance(
            layers[layer_num].intermediate.dense.weight,
            layers[layer_num].intermediate.dense.bias,
            ffn_importance[layer_num],
            target_ffn_dim,
            1
        )
        layers[layer_num].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
        layers[layer_num].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)
        
        # Sort output FFN layer
        weight_sorted, _ = sort_by_importance(
            layers[layer_num].output.dense.weight.transpose(0, 1),
            None,
            ffn_importance[layer_num],
            target_ffn_dim,
            1
        )
        weight_sorted = weight_sorted.transpose(0, 1)
        layers[layer_num].output.dense.weight = torch.nn.Parameter(weight_sorted)
    
    # Update model configuration for the pruned model
    output_name = model_path.split('/')[-2]
    pruned_dir = os.path.join(output_dir, f"{output_name}_{target_num_heads}_{target_ffn_dim}")
    if not os.path.exists(pruned_dir):
        os.makedirs(pruned_dir)
    
    # Update configuration
    model.config.num_attention_heads = target_num_heads
    model.config.intermediate_size = layers[0].intermediate.dense.weight.size(0)
    
    # Save pruned model
    model.config.save_pretrained(pruned_dir)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)
    
    # Evaluate model after pruning on validation set
    print("Evaluating model after pruning on validation set...")
    model = AutoModelForMaskedLM.from_pretrained(pruned_dir, config=pruned_dir+"/config.json", ignore_mismatched_sizes=True)
    model.to(device)
    
    after_metrics = evaluate_mlm_model(model, val_dataset, tokenizer, device)
    print(f"After pruning (validation): {after_metrics}")
    
    # Calculate parameter counts
    orig_model = AutoModelForMaskedLM.from_pretrained(model_path, ignore_mismatched_sizes=True)
    orig_param_count = sum(p.numel() for p in orig_model.parameters())
    pruned_param_count = sum(p.numel() for p in model.parameters())
    
    print(f"Original model parameter count: {orig_param_count:,}")
    print(f"Pruned model parameter count: {pruned_param_count:,}")
    print(f"Reduction: {(1 - pruned_param_count/orig_param_count) * 100:.2f}%")
    
    return {
        "original_model_path": model_path,
        "pruned_model_path": pruned_dir,
        "val_metrics_before": before_metrics,
        "val_metrics_after": after_metrics,
        "param_count_before": orig_param_count,
        "param_count_after": pruned_param_count,
        "reduction_percentage": (1 - pruned_param_count/orig_param_count) * 100
    }

def test_models(original_model_path, pruned_model_path, test_dataset, tokenizer, device="cuda"):
    """Test both original and pruned models on the test set."""
    print("\n=== Testing Models on Test Set ===\n")
    
    print("Loading original model...")
    original_model = AutoModelForMaskedLM.from_pretrained(original_model_path)
    original_model.to(device)
    
    print("Evaluating original model on test set...")
    original_metrics = evaluate_mlm_model(original_model, test_dataset, tokenizer, device)
    print(f"Original model test metrics: {original_metrics}")
    
    print("\nLoading pruned model...")
    pruned_model = AutoModelForMaskedLM.from_pretrained(pruned_model_path, config=pruned_model_path+"/config.json", ignore_mismatched_sizes=True)
    pruned_model.to(device)
    
    print("Evaluating pruned model on test set...")
    pruned_metrics = evaluate_mlm_model(pruned_model, test_dataset, tokenizer, device)
    print(f"Pruned model test metrics: {pruned_metrics}")
    
    print("\n=== Model Comparison on Test Set ===")
    print(f"Original model perplexity: {original_metrics['perplexity']:.4f}")
    print(f"Pruned model perplexity: {pruned_metrics['perplexity']:.4f}")
    print(f"Perplexity difference: {pruned_metrics['perplexity'] - original_metrics['perplexity']:.4f}")
    print(f"Original MLM accuracy: {original_metrics['masked_lm_accuracy']:.4f}")
    print(f"Pruned MLM accuracy: {pruned_metrics['masked_lm_accuracy']:.4f}")
    print(f"MLM accuracy difference: {pruned_metrics['masked_lm_accuracy'] - original_metrics['masked_lm_accuracy']:.4f}")
    
    if torch.cuda.is_available():
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        test_subset = test_dataset.select(range(min(10000, len(test_dataset))))
        
        test_loader = DataLoader(
            test_subset,
            batch_size=8,
            collate_fn=data_collator
        )
        
        for batch in [next(iter(test_loader)) for _ in range(5)]:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                _ = original_model(**batch)
                _ = pruned_model(**batch)
        
        import time
        
        start_time = time.time()
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                _ = original_model(**batch)
        original_time = time.time() - start_time
        
        start_time = time.time()
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                _ = pruned_model(**batch)
        pruned_time = time.time() - start_time
        
        print(f"\nInference time (original): {original_time:.2f}s")
        print(f"Inference time (pruned): {pruned_time:.2f}s")
        print(f"Speedup: {original_time/pruned_time:.2f}x")
    
    return {
        "original_metrics": original_metrics,
        "pruned_metrics": pruned_metrics
    }

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Prune transformer model')
    parser.add_argument('--target_num_heads', type=int, default=12,
                        help='Target number of attention heads (default: 12)')
    parser.add_argument('--target_ffn_dim', type=int, default=2048,
                        help='Target feed-forward network dimension (default: 2048)')
    parser.add_argument('--output_dir', type=str, default='./pruned_models',
                        help='Output directory for pruned models (default: ./pruned_models)')
    parser.add_argument('--model_path', type=str, default="MLRS/mBERTu",
                        help='Path to the base model (default: MLRS/mBERTu)')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to the tokenizer (default: same as model_path)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use for computation (default: cuda if available, else cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--language', type=str, default="mlt_Latn")
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Set tokenizer path to model path if not specified
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    seed = args.seed
    set_seed(seed)
    
    language_code = args.language
    print(f"Loading FLORES-200 dataset for {language_code}...")
    flores_dataset = load_dataset("facebook/flores", language_code)
    flores_val_data = flores_dataset["dev"]
    print(f"FLORES-200 validation data loaded for {language_code}!")
    
    flores_test_data = flores_dataset["devtest"] 
    
    model_path = args.model_path
    
    print("Tokenizing FLORES-200 dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"], 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
    
    tokenized_val_dataset = flores_val_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "id"]
    )
    
    tokenized_test_dataset = flores_test_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "id"]
    )
    
    tokenized_val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    target_num_heads = args.target_num_heads  # Default is usually 12 for base models
    target_ffn_dim = args.target_ffn_dim  # Default is usually 3072 for base models
    output_dir = args.output_dir
    
    print("\n=== Starting Model Pruning with MLM Criterion ===\n")
    pruning_results = prune_model(
        model_path=model_path,
        val_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        target_num_heads=target_num_heads,
        target_ffn_dim=target_ffn_dim,
        output_dir=output_dir,
        device=args.device
    )
    
    print("\n=== Starting Model Testing ===\n")
    test_results = test_models(
        original_model_path=model_path,
        pruned_model_path=pruning_results["pruned_model_path"],
        test_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        device=args.device
    )
    
    print("\n=== Final Summary ===")
    print(f"Original model parameters: {pruning_results['param_count_before']:,}")
    print(f"Pruned model parameters: {pruning_results['param_count_after']:,}")
    print(f"Parameter reduction: {pruning_results['reduction_percentage']:.2f}%")
    print("\nValidation Results:")
    print(f"Original model validation perplexity: {pruning_results['val_metrics_before']['perplexity']:.4f}")
    print(f"Pruned model validation perplexity: {pruning_results['val_metrics_after']['perplexity']:.4f}")
    print(f"Original model validation MLM accuracy: {pruning_results['val_metrics_before']['masked_lm_accuracy']:.4f}")
    print(f"Pruned model validation MLM accuracy: {pruning_results['val_metrics_after']['masked_lm_accuracy']:.4f}")
    print("\nTest Results:")
    print(f"Original model test perplexity: {test_results['original_metrics']['perplexity']:.4f}")
    print(f"Pruned model test perplexity: {test_results['pruned_metrics']['perplexity']:.4f}")
    print(f"Original model test MLM accuracy: {test_results['original_metrics']['masked_lm_accuracy']:.4f}")
    print(f"Pruned model test MLM accuracy: {test_results['pruned_metrics']['masked_lm_accuracy']:.4f}")
    
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_identifier = model_path.split("/")[-2]
    print(model_path)
        
    with open(os.path.join(results_dir, f"{model_identifier}_{target_num_heads}_{target_ffn_dim}.txt"), "w") as f:
        f.write("=== Pruning Configuration ===\n")
        f.write(f"Target number of attention heads: {target_num_heads}\n")
        f.write(f"Target FFN dimension: {target_ffn_dim}\n")
        f.write(f"Original model: {model_path}\n")
        f.write(f"Pruned model: {pruning_results['pruned_model_path']}\n\n")
        
        f.write("=== Parameter Reduction ===\n")
        f.write(f"Original parameters: {pruning_results['param_count_before']:,}\n")
        f.write(f"Pruned parameters: {pruning_results['param_count_after']:,}\n")
        f.write(f"Reduction: {pruning_results['reduction_percentage']:.2f}%\n\n")
        
        f.write("=== Validation Results ===\n")
        f.write(f"Original model perplexity: {pruning_results['val_metrics_before']['perplexity']:.4f}\n")
        f.write(f"Pruned model perplexity: {pruning_results['val_metrics_after']['perplexity']:.4f}\n")
        f.write(f"Original model MLM accuracy: {pruning_results['val_metrics_before']['masked_lm_accuracy']:.4f}\n")
        f.write(f"Pruned model MLM accuracy: {pruning_results['val_metrics_after']['masked_lm_accuracy']:.4f}\n\n")
        
        f.write("=== Test Results ===\n")
        f.write(f"Original model perplexity: {test_results['original_metrics']['perplexity']:.4f}\n")
        f.write(f"Pruned model perplexity: {test_results['pruned_metrics']['perplexity']:.4f}\n")
        f.write(f"Original model MLM accuracy: {test_results['original_metrics']['masked_lm_accuracy']:.4f}\n")
        f.write(f"Pruned model MLM accuracy: {test_results['pruned_metrics']['masked_lm_accuracy']:.4f}\n")
    
    print(f"\nResults saved to {os.path.join(results_dir, f'{model_identifier}_{target_num_heads}_{target_ffn_dim}.txt')}")
    return pruning_results, test_results

if __name__ == "__main__":
    main()