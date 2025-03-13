import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoConfig
import torch
import evaluate
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk
import os

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a smaller student model without distillation")
parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name or path")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the fine-tuned model")
parser.add_argument("--factor", type=int, required=True, help="Layer reduction factor")
parser.add_argument("--parameterization", type=str, choices=["teacher", "random"], required=True, help="Parameterization method ('teacher' or 'random')")
parser.add_argument("--language_code", type=str, default="mlt_Latn", help="FLORES-200 language code for validation")

args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

# Load the teacher model
teacher_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
print("Teacher model loaded!")

# Function to create student model
def create_student_model(layer_reduction_factor, parameterization):
    config = teacher_model.config.to_dict()
    
    # Calculate new number of layers
    original_num_layers = config['num_hidden_layers']
    num_hidden_layers = original_num_layers // layer_reduction_factor
    config['num_hidden_layers'] = num_hidden_layers
    
    # Create student config and model
    student_config = AutoConfig.from_dict(config)
    student_model = AutoModelForMaskedLM(student_config)
    
    if parameterization == 'teacher':
        teacher_state_dict = teacher_model.state_dict()
        student_state_dict = student_model.state_dict()
        new_state_dict = {}

        # Copy non-layer-specific weights
        for key in student_state_dict.keys():
            if 'encoder.layer' not in key and key in teacher_state_dict:
                new_state_dict[key] = teacher_state_dict[key].clone()
        
        # Copy layer weights with stride
        for i in range(num_hidden_layers):
            teacher_layer_idx = i * layer_reduction_factor
            teacher_layer_idx = min(teacher_layer_idx, original_num_layers - 1)
            
            for key in student_state_dict.keys():
                if f"encoder.layer.{i}." in key:
                    teacher_key = key.replace(f"encoder.layer.{i}.", f"encoder.layer.{teacher_layer_idx}.")
                    if teacher_key in teacher_state_dict:
                        new_state_dict[key] = teacher_state_dict[teacher_key].clone()
        
        missing_keys, unexpected_keys = student_model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        print("Parameters successfully copied from teacher")

    elif parameterization == 'random':
        student_model.init_weights()
        print("Parameters randomly initialized")

    else:
        raise ValueError("Invalid parameterization method. Choose 'teacher' or 'random'.")

    return student_model

# Preprocessing and evaluation functions
def preprocess_logits_for_metrics(logits, labels):
    return logits[0].argmax(dim=-1) if isinstance(logits, tuple) else logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    return metric.compute(predictions=preds[mask], references=labels[mask])

# Create and train the student model
print(f"\nTraining student model with {args.factor}x fewer layers and {args.parameterization} parameterization...")
student_model = create_student_model(layer_reduction_factor=args.factor, parameterization=args.parameterization)
print("Student model initialized!")

total_params = count_parameters(student_model)
print(f"Student Model with {args.factor}x fewer layers has {total_params} trainable parameters.")

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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
