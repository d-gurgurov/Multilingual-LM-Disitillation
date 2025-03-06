from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import torch
import evaluate
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk
import json
import os

# Set random seed for reproducibility
torch.manual_seed(42)  

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Function to count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load the Maltese monolingual dataset
dataset = load_from_disk("data_slovak")
print("Data loaded!")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples['content'], truncation=True, max_length=512, padding='max_length', return_special_tokens_mask=True)
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['content', 'warc-target-uri', 'warc-date', 'warc-record-id', 'quality-warnings', 'categories', 'identification-language', 'identification-prob', 'identification-consistency', 'script-percentage', 'num-sents', 'content-length', 'tlsh'])
tokenized_datasets.set_format("torch")
print("Data tokenized!")

# Split the dataset into training and validation sets
val_size = 1000
train_size = len(tokenized_datasets) - val_size
train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

teacher_model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")

# Load the student model
from transformers import BertConfig, BertForMaskedLM

def create_student_model(layer_reduction_factor, parameterization='teacher'):
    # Create a new configuration based on the teacher's config
    config = teacher_model.config.to_dict()
    
    # Calculate the number of layers for the student model
    original_num_layers = config['num_hidden_layers']
    num_hidden_layers = original_num_layers // layer_reduction_factor
    config['num_hidden_layers'] = num_hidden_layers
    
    # Create a proper configuration object for mBERT (BertConfig)
    student_config = BertConfig.from_dict(config)
    
    # Initialize the student model with the modified configuration (using BertForMaskedLM)
    student_model = BertForMaskedLM(student_config)
    
    if parameterization == 'teacher':
        # Get state dictionaries
        teacher_state_dict = teacher_model.state_dict()
        student_state_dict = student_model.state_dict()

        # Create a new state dict for the student model
        new_state_dict = {}
        
        # Copy embedding and other non-layer specific weights
        for key in student_state_dict.keys():
            if 'encoder.layer' not in key:
                if key in teacher_state_dict:
                    new_state_dict[key] = teacher_state_dict[key].clone()
        
        # Copy layer weights with the correct stride
        for i in range(num_hidden_layers):
            teacher_layer_idx = i * layer_reduction_factor
            
            # Make sure we don't exceed the teacher's layer count
            if teacher_layer_idx >= original_num_layers:
                teacher_layer_idx = original_num_layers - 1
            
            # Find all keys for this student layer and copy from corresponding teacher layer
            for key in student_state_dict.keys():
                if f"encoder.layer.{i}." in key:
                    teacher_key = key.replace(f"encoder.layer.{i}.", f"encoder.layer.{teacher_layer_idx}.")
                    if teacher_key in teacher_state_dict:
                        new_state_dict[key] = teacher_state_dict[teacher_key].clone()
        
        # Load the weights into the student model
        missing_keys, unexpected_keys = student_model.load_state_dict(new_state_dict, strict=False)
        
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        if len(missing_keys) > 0:
            print("Sample missing keys:", missing_keys[:3])
        
        print("Parameters successfully copied from teacher")
    
    elif parameterization == 'random':
        # Explicitly reset all parameters to random initialization
        student_model.init_weights()
        print("Parameters randomly initialized")
    
    else:
        raise ValueError(f"Unknown parameterization method: {parameterization}. "
                         "Choose 'teacher' or 'random'.")
    
    return student_model


def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)


# Train models with different layer reduction factors
for factor in [2]:
    print(f"\nTraining student model with {factor}x fewer layers...")
    student_model = create_student_model(layer_reduction_factor=factor, parameterization="teacher")
    print("Student model initialized!")
    
    total_params = count_parameters(student_model)
    print(f"Student Model with {factor}x fewer layers has {total_params} trainable parameters.")
    
    training_args = TrainingArguments(
        output_dir=f'./models/slovak/student_model_{factor}',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        seed=42,
        logging_dir=f'./models/slovak/student_model_{factor}/logs',
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    trainer.train()
