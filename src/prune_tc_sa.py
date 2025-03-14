import os
import torch
import numpy as np
import random
from tqdm import tqdm
from heapq import heappush, heappop
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, SequentialSampler
import logging
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer


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

def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, and F1 score for classification."""
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_model(model, dataset, device, batch_size=8):
    """Evaluate model on a dataset."""
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)
    
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs.loss, outputs.logits
            
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    metrics = compute_metrics(preds, out_label_ids)
    metrics["loss"] = eval_loss
    
    return metrics, preds

def prune_model(model_path, val_dataset, adapter_path, target_num_heads, target_ffn_dim, output_dir, device="cuda"):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoAdapterModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/maltese_model_0.1")
    
    model.load_adapter(adapter_path)
    if "topic_classification" in model.config.prediction_heads:
        model.set_active_adapters("topic_classification")
    elif "sentiment_classification" in model.config.prediction_heads:
        model.set_active_adapters("sentiment_classification")
    else:
        print("Neither 'topic_classification' nor 'sentiment_classification' adapters are available.")
    model.to(device)

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Evaluating model before pruning...")
    before_metrics, _ = evaluate_model(model, val_dataset, device)
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
    
    eval_sampler = SequentialSampler(val_dataset)
    eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=8)
    
    model.eval()
    tot_tokens = 0.0
    
    print("Computing importance scores for pruning...")
    for batch in tqdm(eval_dataloader, desc="Computing importance"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass with head mask to get attention outputs
        outputs = model(output_attentions=True, head_mask=head_mask, **batch)
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
        query_weight = layers[layer_num].attention.self.query.weight
        query_bias = layers[layer_num].attention.self.query.bias
        key_weight = layers[layer_num].attention.self.key.weight
        key_bias = layers[layer_num].attention.self.key.bias
        value_weight = layers[layer_num].attention.self.value.weight
        value_bias = layers[layer_num].attention.self.value.bias
        
        # Sort query, key, value based on importance scores
        query_weight, query_bias = sort_by_importance(
            query_weight, 
            query_bias,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        layers[layer_num].attention.self.query.weight = torch.nn.Parameter(query_weight)
        layers[layer_num].attention.self.query.bias = torch.nn.Parameter(query_bias)
        
        key_weight, key_bias = sort_by_importance(
            key_weight,
            key_bias,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        layers[layer_num].attention.self.key.weight = torch.nn.Parameter(key_weight)
        layers[layer_num].attention.self.key.bias = torch.nn.Parameter(key_bias)
        
        value_weight, value_bias = sort_by_importance(
            value_weight,
            value_bias,
            head_importance[layer_num],
            target_num_heads,
            head_size
        )
        layers[layer_num].attention.self.value.weight = torch.nn.Parameter(value_weight)
        layers[layer_num].attention.self.value.bias = torch.nn.Parameter(value_bias)
        
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
    pruned_dir = os.path.join(output_dir, f"pruned_{target_num_heads}_{target_ffn_dim}")
    if not os.path.exists(pruned_dir):
        os.makedirs(pruned_dir)
    
    # Update configuration
    # model.config.hidden_act = 'relu'  # Use ReLU activation for pruned models
    model.config.num_attention_heads = min(config.num_attention_heads, target_num_heads)
    model.config.intermediate_size = layers[0].intermediate.dense.weight.size(0)
    
    # Save pruned model
    model.config.save_pretrained(pruned_dir)
    model.save_pretrained(pruned_dir)
    tokenizer.save_pretrained(pruned_dir)
    
    # Evaluate model after pruning on validation set
    model = AutoAdapterModel.from_pretrained(pruned_dir, config=pruned_dir+"/config.json", ignore_mismatched_sizes=True)
    # Load the adapter for topic classification
    model.load_adapter(adapter_path)
    if "topic_classification" in model.config.prediction_heads:
        model.set_active_adapters("topic_classification")
    elif "sentiment_classification" in model.config.prediction_heads:
        model.set_active_adapters("sentiment_classification")
    else:
        print("Neither 'topic_classification' nor 'sentiment_classification' adapters are available.")
    
    model.to(device)
    
    print("Evaluating model after pruning on validation set...")
    after_metrics, _ = evaluate_model(model, val_dataset, device)
    print(f"After pruning (validation): {after_metrics}")
    
    # Calculate parameter counts
    orig_model = AutoAdapterModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
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

def test_models(original_model_path, adapter_path, pruned_model_path, test_dataset, device="cuda"):
    """Test both original and pruned models on the test set."""
    print("\n=== Testing Models on Test Set ===\n")
    
    print("Loading original model...")
    original_model = AutoAdapterModel.from_pretrained(original_model_path)
    original_model.load_adapter(adapter_path)
    if "topic_classification" in original_model.config.prediction_heads:
        original_model.set_active_adapters("topic_classification")
    elif "sentiment_classification" in original_model.config.prediction_heads:
        original_model.set_active_adapters("sentiment_classification")
    else:
        print("Neither 'topic_classification' nor 'sentiment_classification' adapters are available.")
    original_model.to(device)
    
    print("Evaluating original model on test set...")
    original_metrics, original_preds = evaluate_model(original_model, test_dataset, device)
    print(f"Original model test metrics: {original_metrics}")
    
    print("\nLoading pruned model...")
    pruned_model = AutoAdapterModel.from_pretrained(pruned_model_path, config=pruned_model_path+"/config.json", ignore_mismatched_sizes=True)
    pruned_model.load_adapter(adapter_path)

    if "topic_classification" in pruned_model.config.prediction_heads:
        pruned_model.set_active_adapters("topic_classification")
    elif "sentiment_classification" in pruned_model.config.prediction_heads:
        pruned_model.set_active_adapters("sentiment_classification")
    else:
        print("Neither 'topic_classification' nor 'sentiment_classification' adapters are available.")

    pruned_model.to(device)
    
    print("Evaluating pruned model on test set...")
    pruned_metrics, pruned_preds = evaluate_model(pruned_model, test_dataset, device)
    print(f"Pruned model test metrics: {pruned_metrics}")
    
    print("\n=== Model Comparison on Test Set ===")
    print(f"Original model accuracy: {original_metrics['accuracy']:.4f}")
    print(f"Pruned model accuracy: {pruned_metrics['accuracy']:.4f}")
    print(f"Accuracy difference: {pruned_metrics['accuracy'] - original_metrics['accuracy']:.4f}")
    print(f"F1 score difference: {pruned_metrics['f1'] - original_metrics['f1']:.4f}")
    
    if torch.cuda.is_available():
        for _ in range(5):
            batch = {k: test_dataset[0:8][k].to(device) for k in test_dataset[0:8]}
            with torch.no_grad():
                _ = original_model(**batch)
                _ = pruned_model(**batch)
                
        import time
        
        start_time = time.time()
        for i in range(0, min(len(test_dataset), 1000), 8):
            end_idx = min(i + 8, len(test_dataset))
            batch = {k: test_dataset[i:end_idx][k].to(device) for k in test_dataset[i:end_idx]}
            with torch.no_grad():
                _ = original_model(**batch)
        original_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(0, min(len(test_dataset), 1000), 8):
            end_idx = min(i + 8, len(test_dataset))
            batch = {k: test_dataset[i:end_idx][k].to(device) for k in test_dataset[i:end_idx]}
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    def find_latest_checkpoint(model_base_path):
        if not os.path.exists(model_base_path):
            raise ValueError(f"Model path {model_base_path} does not exist")
            
        checkpoints = [d for d in os.listdir(model_base_path) if d.startswith("checkpoint")]
        if not checkpoints:
            return model_base_path
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        return os.path.join(model_base_path, checkpoints[-1])
    
    print("Loading dataset...")
    dataset_tc = load_dataset('Davlan/sib200', 'mlt_Latn')
    dataset_sa = load_dataset('DGurgurov/maltese_sa')

    categories = dataset_tc["train"].unique("category")
    category_to_id = {category: idx for idx, category in enumerate(categories)}
    
    def encode_category(example):
        example["category"] = category_to_id[example["category"]]
        return example
    
    dataset_tc = dataset_tc.map(encode_category)

    categories = dataset_sa["train"].unique("label")
    category_to_id = {category: idx for idx, category in enumerate(categories)}
    
    def encode_category(example):
        example["label"] = category_to_id[example["label"]]
        return example

    dataset_sa = dataset_sa.map(encode_category)

    model_base_path = "/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/maltese_model_0.1"
    try:
        model_path = find_latest_checkpoint(model_base_path)
        print(f"Using model from: {model_path}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Using default path instead")
        model_path = "MLRS/mBERTu" 
    
    tokenizer = AutoTokenizer.from_pretrained("/netscratch/dgurgurov/projects2025/distillation_mbert/reduce_vocab/maltese_model_0.1")
    
    print("Tokenizing dataset...")
    def encode_batch(examples):
        encoded = tokenizer(
            examples["text"], 
            max_length=256, 
            truncation=True, 
            padding="max_length"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": examples.get("category", examples.get("label"))
        }
    
    encoded_dataset = dataset_sa.map(encode_batch, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    target_num_heads = 12  # Default is usually 12 for base models
    target_ffn_dim = 2560 - 512  # Default is usually 3072 for base models
    output_dir = "./pruned_models_sa"
    seed = 42
    
    set_seed(seed)

    adapter_path = "/netscratch/dgurgurov/projects2025/distillation_mbert/sa/distilled_mbertu_2_double_loss_alpha_0.5_mse_unsoft/1/checkpoint-760/sentiment_classification"
    
    print("\n=== Starting Model Pruning ===\n")
    pruning_results = prune_model(
        model_path=model_path,
        adapter_path = adapter_path,
        val_dataset=encoded_dataset["train"],
        target_num_heads=target_num_heads,
        target_ffn_dim=target_ffn_dim,
        output_dir=output_dir,
        device=device
    )
    
    print("\n=== Starting Model Testing ===\n")
    test_results = test_models(
        original_model_path=model_path,
        adapter_path = adapter_path,
        pruned_model_path=pruning_results["pruned_model_path"],
        test_dataset=encoded_dataset["test"],
        device=device
    )
    
    print("\n=== Final Summary ===")
    print(f"Original model parameters: {pruning_results['param_count_before']:,}")
    print(f"Pruned model parameters: {pruning_results['param_count_after']:,}")
    print(f"Parameter reduction: {pruning_results['reduction_percentage']:.2f}%")
    print("\nValidation Results:")
    print(f"Original model validation accuracy: {pruning_results['val_metrics_before']['accuracy']:.4f}")
    print(f"Pruned model validation accuracy: {pruning_results['val_metrics_after']['accuracy']:.4f}")
    print("\nTest Results:")
    print(f"Original model test accuracy: {test_results['original_metrics']['accuracy']:.4f}")
    print(f"Pruned model test accuracy: {test_results['pruned_metrics']['accuracy']:.4f}")
    print(f"Original model test F1: {test_results['original_metrics']['f1']:.4f}")
    print(f"Pruned model test F1: {test_results['pruned_metrics']['f1']:.4f}")
    
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    with open(os.path.join(results_dir, f"pruning_results_{target_num_heads}_{target_ffn_dim}.txt"), "w") as f:
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
        f.write(f"Original model: {pruning_results['val_metrics_before']}\n")
        f.write(f"Pruned model: {pruning_results['val_metrics_after']}\n\n")
        
        f.write("=== Test Results ===\n")
        f.write(f"Original model: {test_results['original_metrics']}\n")
        f.write(f"Pruned model: {test_results['pruned_metrics']}\n")
    
    print(f"\nResults saved to {os.path.join(results_dir, f'pruning_results_{target_num_heads}_{target_ffn_dim}.txt')}")
    return pruning_results, test_results

if __name__ == "__main__":
    main()