import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, XLMRobertaConfig, BertConfig
import torch
import evaluate
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk, load_dataset
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
parser.add_argument("--layer_selection", type=str, default="stride", help="Layer selection for student initialization")
parser.add_argument("--student_model_path", type=str, default=None, help="Student model path")
parser.add_argument("--tiny_model_init", type=str, default=None, help="Tiny model init")
parser.add_argument("--hidden_size", type=int, default=312, help="Tiny model hidden size")
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
tokenized_val_dataset = tokenized_val_dataset.remove_columns([col for col in flores_val_data.column_names if col != 'sentence']).remove_columns(['sentence'])
tokenized_val_dataset.set_format("torch")
print("Validation data tokenized!")

# Load the teacher model
teacher_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
print("Teacher model loaded!")

def transfer_weights_by_svd(original_model, modified_model):
    """
    Transfer weights from original model to modified model using SVD for dimensionality reduction.
    
    Args:
        original_model: Source BERT model
        modified_model: Target BERT model with reduced dimensions
        
    Returns:
        modified_model: Model with transferred weights
    """
    # Get the dimension reduction parameters
    orig_hidden_size = original_model.config.hidden_size
    new_hidden_size = modified_model.config.hidden_size
    orig_intermediate_size = original_model.config.intermediate_size
    new_intermediate_size = modified_model.config.intermediate_size
    
    # Get state dictionaries
    orig_state_dict = original_model.state_dict()
    mod_state_dict = modified_model.state_dict()
    
    # Process each parameter
    for name, param in orig_state_dict.items():
        # Skip if parameter doesn't exist in modified model
        if name not in mod_state_dict:
            print(f"Skipping parameter {name} as it doesn't exist in modified model")
            continue
        
        # Get the target parameter shape
        target_shape = mod_state_dict[name].shape
        
        # Check tensor dimensionality
        if len(param.shape) < 2:
            # For 1D tensors (biases, etc.), just truncate or pad as needed
            if len(target_shape) == 1:
                if target_shape[0] <= param.shape[0]:
                    # Truncate
                    mod_state_dict[name] = param[:target_shape[0]]
                else:
                    # Pad (unlikely scenario)
                    mod_state_dict[name] = torch.nn.functional.pad(param, (0, target_shape[0] - param.shape[0]))
            else:
                print(f"Dimension mismatch for {name}: cannot convert 1D to {len(target_shape)}D")
            continue
            
        # Handle embeddings (keep all vocab but reduce embedding dimension)
        if 'embeddings.word_embeddings.weight' in name or 'embeddings.position_embeddings.weight' in name or 'embeddings.token_type_embeddings.weight' in name:
            # For embeddings, we need to reduce the embedding dimension
            u, s, vh = torch.linalg.svd(param, full_matrices=False)
            # Ensure we don't exceed the available singular values
            k = min(new_hidden_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
        
        # Handle attention query, key, value weights
        elif any(x in name for x in ['query', 'key', 'value']) and 'weight' in name:
            u, s, vh = torch.linalg.svd(param, full_matrices=False)
            k = min(new_hidden_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
        
        # Handle attention output projection
        elif 'attention.output.dense.weight' in name:
            u, s, vh = torch.linalg.svd(param, full_matrices=False)
            k = min(new_hidden_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
        
        # Handle intermediate layer projection
        elif 'intermediate.dense.weight' in name:
            u, s, vh = torch.linalg.svd(param, full_matrices=False)
            k = min(new_intermediate_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
        
        # Handle output layer projection
        elif 'output.dense.weight' in name and not 'attention' in name:
            u, s, vh = torch.linalg.svd(param, full_matrices=False)
            k = min(new_hidden_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
        
        # Handle classifier/MLM head
        elif 'cls' in name and 'weight' in name and not 'predictions.decoder' in name:
            u, s, vh = torch.linalg.svd(param, full_matrices=False)
            k = min(new_hidden_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
        
        # Handle decoder projection (vocabulary projection)
        elif 'predictions.decoder.weight' in name:
            # For decoder, we need to reduce the input dimension but keep the vocab size
            u, s, vh = torch.linalg.svd(param.T, full_matrices=False)
            k = min(new_hidden_size, len(s))
            reduced_param = torch.matmul(u[:, :k], torch.diag(s[:k]) @ vh[:k, :])
            mod_state_dict[name] = reduced_param.T[:target_shape[0], :target_shape[1]]
        
        # Handle biases and LayerNorm parameters (direct copy or truncate)
        elif 'bias' in name or 'LayerNorm' in name:
            if param.shape == target_shape:
                mod_state_dict[name] = param.clone()
            else:
                # For 1D tensors, just truncate
                if len(param.shape) == 1 and len(target_shape) == 1:
                    mod_state_dict[name] = param[:target_shape[0]]
                else:
                    print(f"Complex shape mismatch for {name}: {param.shape} vs {target_shape}")
        
        # Handle any other parameters - fallback to trying to match shapes
        else:
            if param.shape == target_shape:
                mod_state_dict[name] = param.clone()
            else:
                print(f"Shape mismatch for {name}: {param.shape} vs {target_shape} - attempting reshape")
                try:
                    if len(param.shape) == 2:
                        # For 2D matrices, use SVD
                        u, s, vh = torch.linalg.svd(param, full_matrices=False)
                        min_dim = min(min(target_shape), len(s))
                        reduced_param = torch.matmul(u[:, :min_dim], torch.diag(s[:min_dim]) @ vh[:min_dim, :])
                        mod_state_dict[name] = reduced_param[:target_shape[0], :target_shape[1]]
                    else:
                        # For other shapes, try to truncate
                        slices = tuple(slice(0, dim) for dim in target_shape)
                        mod_state_dict[name] = param[slices].clone()
                except Exception as e:
                    print(f"Failed to transfer parameter {name} with shape {param.shape}: {e}")
    
    # Load the modified state dict
    modified_model.load_state_dict(mod_state_dict)
    
    return modified_model

def transfer_weights_by_truncation(original_model, modified_model):
        """
        Transfer weights from original model to modified model by simple truncation.
        Takes the first k weights/units from each layer without SVD.
        
        Args:
            original_model: Source BERT model
            modified_model: Target BERT model with reduced dimensions
            
        Returns:
            modified_model: Model with transferred weights
        """
        # Get state dictionaries
        orig_state_dict = original_model.state_dict()
        mod_state_dict = modified_model.state_dict()
        
        # Process each parameter
        for name, param in orig_state_dict.items():
            # Skip if parameter doesn't exist in modified model
            if name not in mod_state_dict:
                print(f"Skipping parameter {name} as it doesn't exist in modified model")
                continue
            
            # Get the target parameter shape
            target_shape = mod_state_dict[name].shape
            
            # Simple truncation approach
            try:
                if len(param.shape) == 1:
                    # For 1D tensors (biases, etc.), just take the first k elements
                    mod_state_dict[name] = param[:target_shape[0]].clone()
                
                elif len(param.shape) == 2:
                    # For 2D matrices, truncate both dimensions
                    mod_state_dict[name] = param[:target_shape[0], :target_shape[1]].clone()
                
                else:
                    # For higher dimension tensors, use slicing
                    slices = tuple(slice(0, dim) for dim in target_shape)
                    mod_state_dict[name] = param[slices].clone()
                    
            except Exception as e:
                print(f"Failed to transfer parameter {name} with shape {param.shape}: {e}")
        
        # Load the modified state dict
        modified_model.load_state_dict(mod_state_dict)
        
        return modified_model

from transformers import XLMRobertaForMaskedLM, BertForMaskedLM, XLMRobertaConfig, BertConfig

def create_student_model(layer_reduction_factor, parameterization='teacher', layer_selection='stride', student_model_path=args.student_model_path, tiny_model_init=args.tiny_model_init):
    # Create a new configuration based on the teacher's config
    config = teacher_model.config.to_dict()

    # Calculate the number of layers for the student model
    original_num_layers = config['num_hidden_layers']
    num_hidden_layers = original_num_layers // layer_reduction_factor
    config['num_hidden_layers'] = num_hidden_layers

    # Create student config and model
    if "xlm-r" in str(args.model_name).lower():
        student_config = XLMRobertaConfig.from_dict(config)
        student_model = XLMRobertaForMaskedLM(student_config)
        print("Using XLM-R config!")
    elif "bert" in str(args.model_name).lower():
        student_config = BertConfig.from_dict(config)
        student_model = BertForMaskedLM(student_config)
        print("Using BERT config!")
    else:
        raise ValueError("Unknown model type. Ensure model_name contains 'bert' or 'xlm'.")

    if parameterization == 'teacher':
        # Get state dictionaries
        teacher_state_dict = teacher_model.state_dict()
        student_state_dict = student_model.state_dict()

        # Create a new state dict for the student model
        new_state_dict = {}

        # Copy embedding and other non-layer-specific weights
        for key in student_state_dict.keys():
            if 'encoder.layer' not in key:
                if key in teacher_state_dict:
                    new_state_dict[key] = teacher_state_dict[key].clone()

        # Select layers based on the chosen method
        if layer_selection == 'stride':
            selected_layers = [i * layer_reduction_factor for i in range(num_hidden_layers)]
        elif layer_selection == 'first_k':
            selected_layers = list(range(num_hidden_layers))
        elif layer_selection == 'last_k':
            selected_layers = list(range(original_num_layers - num_hidden_layers, original_num_layers))
        elif layer_selection == 'first_last_k':
            half_k = num_hidden_layers // 2
            selected_layers = list(range(half_k)) + list(range(original_num_layers - half_k, original_num_layers))
        else:
            raise ValueError(f"Unknown layer selection method: {layer_selection}. Choose 'stride', 'first_k', or 'last_k'.")

        # Copy layer weights
        for i, teacher_layer_idx in enumerate(selected_layers):
            # Ensure we don't exceed the teacher's layer count
            teacher_layer_idx = min(teacher_layer_idx, original_num_layers - 1)

            for key in student_state_dict.keys():
                if f"encoder.layer.{i}." in key:
                    teacher_key = key.replace(f"encoder.layer.{i}.", f"encoder.layer.{teacher_layer_idx}.")
                    if teacher_key in teacher_state_dict:
                        new_state_dict[key] = teacher_state_dict[teacher_key].clone()

        # Load the weights into the student model
        missing_keys, unexpected_keys = student_model.load_state_dict(new_state_dict, strict=False)

        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        if missing_keys:
            print("Sample missing keys:", missing_keys[:3])

        print("Parameters successfully copied from teacher")

    elif parameterization == 'random':
        # Explicitly reset all parameters to random initialization
        student_model.init_weights()
        print("Parameters randomly initialized")

    else:
        raise ValueError(f"Unknown parameterization method: {parameterization}. Choose 'teacher' or 'random'.")

    def modify_bert_model(model, hidden_size=312, intermediate_size=2048, num_attention_heads=12):
        config = model.config
        
        # Update model config
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.num_attention_heads = num_attention_heads  
        
        # Ensure hidden_size is divisible by num_attention_heads
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads!"
        
        # Reinitialize model with modified config
        new_model = AutoModelForMaskedLM.from_config(config)
        
        return new_model

    if student_model_path is not None:
        hidden_size=args.hidden_size
        intermediate_size=2048
        student_model_path = AutoModelForMaskedLM.from_pretrained(student_model_path)
        modified_model = modify_bert_model(student_model_path, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=12)
        if tiny_model_init == "truncation":
            student_model = transfer_weights_by_truncation(student_model_path, modified_model)
        elif tiny_model_init == "svd":
            student_model = transfer_weights_by_svd(student_model_path, modified_model)
        
        print(f"Using the modified model with smaller hidden ({hidden_size}) and intermediate ({intermediate_size}) sizes")
        print(f"Using {tiny_model_init} initialization for hidden size")
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
student_model = create_student_model(layer_reduction_factor=args.factor, parameterization=args.parameterization, layer_selection=args.layer_selection)
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
