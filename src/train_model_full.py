import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import evaluate
import torch
from datasets import load_from_disk
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a masked language model")
parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name or path")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Load the dataset
dataset = load_from_disk(args.dataset_path)
print("Data loaded!")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['content'], truncation=True, max_length=512, padding='max_length', return_special_tokens_mask=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['content', 'warc-target-uri', 'warc-date', 'warc-record-id', 'quality-warnings', 'categories', 'identification-language', 'identification-prob', 'identification-consistency', 'script-percentage', 'num-sents', 'content-length', 'tlsh'])
tokenized_datasets.set_format("torch")
print("Data tokenized!")

# Split the dataset into training and validation sets
val_size = 1000
train_size = len(tokenized_datasets) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(tokenized_datasets, [train_size, val_size])

# Load the model
model = AutoModelForMaskedLM.from_pretrained(args.model_name)
print("Model loaded!")

# Define evaluation metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    seed=42,
    logging_dir=args.output_dir + "/logs",
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=1,
    logging_steps=100,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# Train the model
trainer.train()
