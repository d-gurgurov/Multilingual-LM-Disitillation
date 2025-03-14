import os
import json
import numpy as np
import torch
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed value")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer (defaults to model_path if not specified)")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
parser.add_argument("--dataset", type=str, default="DGurgurov/maltese_sa", help="Dataset name to use (default: DGurgurov/maltese_sa)")
parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
args = parser.parse_args()

# Set random seed
seed_value = args.seed
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, TrainingArguments
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer
from adapters.composition import Stack
import evaluate

def find_latest_checkpoint(model_base_path: str) -> str:
    """Finds the latest checkpoint directory for the given language."""
    checkpoints = [d for d in os.listdir(model_base_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    return os.path.join(model_base_path, checkpoints[-1])

# Load dataset for sentiment analysis
dataset = load_dataset(args.dataset)
categories = dataset["train"].unique("label")
category_to_id = {category: idx for idx, category in enumerate(categories)}

def encode_category(example):
    example["label"] = category_to_id[example["label"]]
    return example

dataset = dataset.map(encode_category)

# Use tokenizer path if provided, otherwise use model path
tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path

def encode_batch(examples):
    """Encodes a batch of input data using the model tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text, label in zip(examples["text"], examples["label"]):
        encoded = tokenizer(text, max_length=args.max_length, truncation=True, padding="max_length")
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
        all_encoded["labels"].append(label)
    
    return all_encoded

dataset = dataset.map(encode_batch, batched=True)
dataset.set_format(columns=["input_ids", "attention_mask", "labels"])

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Load model and adapter
model_path = find_latest_checkpoint(args.model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoAdapterModel.from_pretrained(model_path, config=config)
print("Original number of model parameters:", model.num_parameters())

# Add classification head and set active adapter
adapter_name = "sentiment_analysis"
model.add_adapter(adapter_name)
num_labels = len(categories)
model.add_classification_head(adapter_name, num_labels=num_labels)
model.train_adapter([adapter_name])

# Create output directory
model_name = os.path.basename(args.model_path)
output_dir = os.path.join(args.output_dir, model_name, str(seed_value))
os.makedirs(output_dir, exist_ok=True)

print(f"Model path: {model_path}")
print(f"Dataset: {args.dataset}")
print(f"Number of labels: {num_labels}")
print(f"Output directory: {output_dir}")
print(f"Seed value: {seed_value}")

training_args = TrainingArguments(
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    logging_steps=20,
)

f1_metric = evaluate.load("f1")
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda pred: {"f1": f1_metric.compute(predictions=np.argmax(pred.predictions, axis=-1), references=pred.label_ids, average="macro")["f1"]},
)

trainer.train()

# Evaluate on test set
test_predictions = trainer.predict(test_dataset)
test_f1 = f1_metric.compute(predictions=np.argmax(test_predictions.predictions, axis=-1), references=test_predictions.label_ids, average="macro")["f1"]
print("Test F1 score:", test_f1)

# Save metrics
output_file_path = os.path.join(output_dir, "f1.json")
with open(output_file_path, "w") as json_file:
    json.dump({"f1": test_f1}, json_file, indent=2)

# Save configuration
config_file_path = os.path.join(output_dir, "config.json")
with open(config_file_path, "w") as json_file:
    config_dict = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "dataset": args.dataset,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seed": seed_value,
        "num_labels": num_labels
    }
    json.dump(config_dict, json_file, indent=2)