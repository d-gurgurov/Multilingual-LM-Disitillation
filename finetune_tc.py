import os
import json
import numpy as np
import torch
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed value")
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
    return os.path.join(local_model_path, checkpoints[-1])

# Load dataset for topic classification in mlt_Latn
dataset = load_dataset('Davlan/sib200', 'mlt_Latn')
categories = dataset["train"].unique("category")
category_to_id = {category: idx for idx, category in enumerate(categories)}

def encode_category(example):
    example["category"] = category_to_id[example["category"]]
    return example

dataset = dataset.map(encode_category)

def encode_batch(examples):
    """Encodes a batch of input data using the model tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text, label in zip(examples["text"], examples["category"]):
        encoded = tokenizer(text, max_length=256, truncation=True, padding="max_length")
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
local_model_path = "models/distilled_model_2_double_loss_alpha_0.8_mse_unsoft"
model_path = find_latest_checkpoint(local_model_path)
model_path = "google-bert/bert-base-multilingual-cased"
config = AutoConfig.from_pretrained(model_path)
model = AutoAdapterModel.from_pretrained(model_path, config=config)

# Add classification head and set active adapter
model.add_adapter("topic_classification")
model.add_classification_head("topic_classification", num_labels=7)
model.train_adapter(["topic_classification"])

save_path = local_model_path.split('/')[-1]
save_path = "mbert"
print(f"Local model path: {save_path}")
print(f"Seed value: {seed_value}")

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    output_dir=f"./tc/{save_path}/{seed_value}",
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
output_file_path = f"./tc/{save_path}/{seed_value}/f1.json"
with open(output_file_path, "w") as json_file:
    json.dump({"f1": test_f1}, json_file, indent=2)
