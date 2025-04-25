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
parser.add_argument("--language", type=str, default="mlt_Latn", help="Language code for dataset (default: mlt_Latn)")
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
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer, BnConfig
from adapters.composition import Stack
from typing import List, Literal, Optional, Union
from collections.abc import Mapping
import evaluate

def find_latest_checkpoint(model_path: str) -> str:
    """
    Finds the latest checkpoint directory within the given model folder.
    """
    # Try loading from Hugging Face if not found locally
    print(f"Model not found locally. Attempting to load from Hugging Face: {model_path}")
    try:
        AutoAdapterModel.from_pretrained(model_path)
        model = model_path
        return model
    except Exception as e:
        print(f"Model not found in local directories or Hugging Face: {e}")

    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
    if not checkpoints:
        return model_path  # If no checkpoint folders, assume model is in base path
    
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    return os.path.join(model_path, checkpoints[-1])  # Return latest checkpoint path
   
# Load dataset for topic classification
dataset = load_dataset('Davlan/sib200', args.language)
categories = dataset["train"].unique("category")
category_to_id = {category: idx for idx, category in enumerate(categories)}

def encode_category(example):
    example["category"] = category_to_id[example["category"]]
    return example

dataset = dataset.map(encode_category)

# Load model and adapter
model_path = find_latest_checkpoint(args.model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoAdapterModel.from_pretrained(model_path, config=config)
print("Original number of model parameters:", model.num_parameters())

# Use tokenizer path if provided, otherwise use model path
tokenizer_path = args.tokenizer_path if args.tokenizer_path else model_path

def encode_batch(examples):
    """Encodes a batch of input data using the model tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text, label in zip(examples["text"], examples["category"]):
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

adapter_name = f"topic_classification_{args.language}"

from dataclasses import dataclass

@dataclass(eq=False)
class SeqBnConfig(BnConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """
    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 4

    def to_dict(self):
        # Convert Mapping to dict if necessary
        reduction_factor = (
            dict(self.reduction_factor) if isinstance(self.reduction_factor, Mapping) else self.reduction_factor
        )
        return {
            "original_ln_before": self.original_ln_before,
            "original_ln_after": self.original_ln_after,
            "residual_before_ln": self.residual_before_ln,
            "adapter_residual_before_ln": self.adapter_residual_before_ln,
            "ln_before": self.ln_before,
            "ln_after": self.ln_after,
            "mh_adapter": self.mh_adapter,
            "output_adapter": self.output_adapter,
            "non_linearity": self.non_linearity,
            "reduction_factor": reduction_factor,
        }

model.add_adapter(adapter_name, SeqBnConfig())

num_labels = len(categories)
model.add_classification_head(adapter_name, num_labels=num_labels)

model.train_adapter([adapter_name])

# Convert adapter config to a dict manually
print(model.adapters_config.config_map)
for adapter, obj in model.adapters_config.config_map.items():
    adapter_config = model.adapters_config.config_map[adapter]
    model.adapters_config.config_map[adapter] = adapter_config.to_dict()

print(model.adapter_summary())


# Create output directory
model_name = os.path.basename(args.model_path)
output_dir = os.path.join(args.output_dir, model_name, str(seed_value))
os.makedirs(output_dir, exist_ok=True)

print(f"Model path: {model_path}")
print(f"Language: {args.language}")
print(f"Number of categories: {num_labels}")
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
        "language": args.language,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seed": seed_value,
        "num_categories": num_labels
    }
    json.dump(config_dict, json_file, indent=2)