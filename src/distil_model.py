from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import evaluate
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk, load_dataset
import json
import argparse
import os

# Set random seed for reproducibility
torch.manual_seed(42)  

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Argument parser
parser = argparse.ArgumentParser(description="Distillation Training Script")

parser.add_argument("--model_name", type=str, required=True, help="Teacher model name")
parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--factor", type=int, required=True, help="Layer reduction factor")
parser.add_argument("--parameterization", type=str, choices=["teacher", "random"], required=True, help="Parameterization method")
parser.add_argument("--alpha", type=float, default=0.5, help="Distillation loss weight")
parser.add_argument("--temperature", type=float, default=0.01, help="Softmax temperature")
parser.add_argument("--teacher_loss", type=str, choices=["KL", "KL_unsoft", "MSE"], default="MSE", help="Teacher loss type")
parser.add_argument("--language_code", type=str, default="mlt_Latn", help="FLORES-200 language code for validation")

args = parser.parse_args()

# Define the custom TrainingArguments class
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, teacher_loss="KD", **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher_loss = teacher_loss

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._move_model_to_device(self.teacher, self.device)
        self.teacher.eval()
        
        # Additional tracking for losses
        self.student_loss_tracking = []
        self.teacher_loss_tracking = []
        self.total_loss_tracking = []

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute student outputs and loss
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        
        # Compute teacher outputs with no gradient
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        
        # Ensure logits are of the same size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Loss calculation
        if self.args.teacher_loss == "KL":
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (loss_function(
                    F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                    F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        elif self.args.teacher_loss == "KL_unsoft":
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = loss_function(
                F.log_softmax(outputs_student.logits, dim=-1),
                F.softmax(outputs_teacher.logits, dim=-1)
            )
        elif self.args.teacher_loss == "MSE":
            loss_function = nn.MSELoss(reduction="mean")
            loss_logits = loss_function(
                outputs_student.logits, 
                outputs_teacher.logits
            )

        else:
            raise ValueError(f"Unsupported loss type: {teacher_loss}")

        # Total loss computation
        total_loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits

        # Track losses for logging
        if self.state.global_step % self.args.logging_steps == 0:
            self.student_loss_tracking.append(student_loss.item())
            self.teacher_loss_tracking.append(loss_logits.item())
            self.total_loss_tracking.append(total_loss.item())

        return (total_loss, outputs_student) if return_outputs else total_loss

    def compute_eval_loss(self, model, inputs):
        # Compute student outputs and loss
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        
        # Compute teacher outputs with no gradient
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        
        # Ensure logits are of the same size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Loss calculation
        if self.args.teacher_loss == "KL":
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (loss_function(
                    F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                    F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        elif self.args.teacher_loss == "KL_unsoft":
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = loss_function(
                F.log_softmax(outputs_student.logits, dim=-1),
                F.softmax(outputs_teacher.logits, dim=-1)
            )
        elif self.args.teacher_loss == "MSE":
            loss_function = nn.MSELoss(reduction="mean")
            loss_logits = loss_function(
                outputs_student.logits, 
                outputs_teacher.logits
            )
        else:
            raise ValueError(f"Unsupported loss type: {teacher_loss}")
        
        # Total loss computation
        total_loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits

        # Track losses for logging
        self.student_loss_tracking.append(student_loss.item())
        self.teacher_loss_tracking.append(loss_logits.item())
        self.total_loss_tracking.append(total_loss.item())

        return total_loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Reset tracking lists before evaluation
        self.student_loss_tracking = []
        self.teacher_loss_tracking = []
        self.total_loss_tracking = []
        
        # Perform the standard evaluation loop
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        
        # Compute average losses if any were tracked
        if self.student_loss_tracking:
            output.metrics[f'{metric_key_prefix}/student_loss'] = sum(self.student_loss_tracking) / len(self.student_loss_tracking)
            output.metrics[f'{metric_key_prefix}/teacher_loss'] = sum(self.teacher_loss_tracking) / len(self.teacher_loss_tracking)
            output.metrics[f'{metric_key_prefix}/total_loss'] = sum(self.total_loss_tracking) / len(self.total_loss_tracking)
        
        return output

    def training_step(self, model, inputs):
        # Reset tracking lists before each training step
        if self.state.global_step % self.args.logging_steps == 0:
            self.student_loss_tracking = []
            self.teacher_loss_tracking = []
            self.total_loss_tracking = []
        
        # Perform the training step
        loss = super().training_step(model, inputs)
        
        # Log losses at logging steps
        if self.state.global_step % self.args.logging_steps == 0:
            if self.student_loss_tracking:
                self.log({
                    'student_loss': sum(self.student_loss_tracking) / len(self.student_loss_tracking),
                    'teacher_loss': sum(self.teacher_loss_tracking) / len(self.teacher_loss_tracking),
                    'total_loss': sum(self.total_loss_tracking) / len(self.total_loss_tracking)
                })
        
        return loss

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
tokenized_val_dataset = tokenized_val_dataset.remove_columns([col for col in flores_val_data.column_names if col != 'sentence']).remove_columns(['sentence'])
tokenized_val_dataset.set_format("torch")
print("Validation data tokenized!")

metric = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

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

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['content', 'warc-target-uri', 'warc-date', 'warc-record-id', 'quality-warnings', 'categories', 'identification-language', 'identification-prob', 'identification-consistency', 'script-percentage', 'num-sents', 'content-length', 'tlsh'])
tokenized_datasets.set_format("torch")
print("Data tokenized!")

# Split the dataset into training and validation sets
val_size = 1000
train_size = len(tokenized_datasets) - val_size
train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size])

# Load the teacher model
teacher_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
teacher_model.eval()
print("Teacher model loaded!")

def create_student_model(layer_reduction_factor, parameterization='teacher'):
    # Create a new configuration based on the teacher's config
    config = teacher_model.config.to_dict()
    
    # Calculate the number of layers for the student model
    original_num_layers = config['num_hidden_layers']
    num_hidden_layers = original_num_layers // layer_reduction_factor
    config['num_hidden_layers'] = num_hidden_layers
    
    # Create a proper configuration object for mBERT (BertConfig)
    student_config = AutoConfig.from_dict(config)
    student_model = AutoModelForMaskedLM(student_config)
    
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

# Distill models with different layer reduction factors
for factor in [args.factor]:
    print(f"\nDistilling model with {factor}x fewer layers...")
    student_model = create_student_model(layer_reduction_factor=factor, parameterization=args.parameterization)
    print("Student model loaded!")

    # Count and print the number of parameters in the student model
    total_params = count_parameters(student_model)
    teacher_params = count_parameters(teacher_model)
    print(f"Student Model with {factor}x fewer layers has {total_params} trainable parameters.")
    print(f"Teacher Model has {teacher_params} parameters.")

    # Create the training arguments for distillation
    distill_training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        seed=42,
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        alpha=args.alpha,  # Distillation factor
        temperature=args.temperature,  # Softening temperature
        teacher_loss=args.teacher_loss,
        save_total_limit=1,
        logging_steps=100,
    )

    # data_collator #
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Create a DistillationTrainer instance
    distill_trainer = DistillationTrainer(
        model=student_model,
        args=distill_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        teacher_model=teacher_model,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Distill the model
    distill_trainer.train()