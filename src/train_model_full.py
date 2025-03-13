import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import evaluate
import torch
from datasets import load_from_disk, load_dataset
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a masked language model")
parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name or path")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
parser.add_argument("--language_code", type=str, default="mlt_Latn", help="FLORES-200 language code for validation")
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Load the training dataset
train_dataset = load_from_disk(args.dataset_path)
print("Training data loaded!")

# Load FLORES-200 for validation (Maltese by default)
flores_dataset = load_dataset("facebook/flores", name=args.language_code)
flores_val_data = flores_dataset["dev"]  # Use the dev split
print(f"FLORES-200 validation data loaded for {args.language_code}!")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# Tokenize the training dataset
def tokenize_function(examples):
    return tokenizer(examples['content'], truncation=True, max_length=512, padding='max_length', return_special_tokens_mask=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['content', 'warc-target-uri', 'warc-date', 'warc-record-id', 'quality-warnings', 'categories', 'identification-language', 'identification-prob', 'identification-consistency', 'script-percentage', 'num-sents', 'content-length', 'tlsh'])
tokenized_train_dataset.set_format("torch")
print("Training data tokenized!")

# Tokenize the FLORES-200 validation dataset
def tokenize_flores(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=512, padding='max_length', return_special_tokens_mask=True)

tokenized_val_dataset = flores_val_data.map(tokenize_flores, batched=True)
tokenized_val_dataset = tokenized_val_dataset.remove_columns([col for col in flores_val_dataset.column_names if col != 'sentence']).remove_columns(['sentence'])
tokenized_val_dataset.set_format("torch")
print("Validation data tokenized!")

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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# Train the model
trainer.train()