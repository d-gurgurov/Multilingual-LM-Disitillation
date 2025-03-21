import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import hashlib
import matplotlib as mpl

from scipy.ndimage import gaussian_filter1d

# Path to the parent folder containing the model folders
base_folder = "/netscratch/dgurgurov/projects2025/distillation_mbert/tiny_models/maltese"

# Extract language name from base_folder
language_name = os.path.basename(base_folder).capitalize()

# Parameter to control plotting frequency (plot every n points)
plot_every_n = 1 

# Function to extract loss and accuracy from TensorBoard logs
def extract_logs(log_dir):
    train_loss, val_loss, val_acc = [], [], []
    train_steps, val_steps = [], []
    student_loss, teacher_loss = [], []
    eval_student_loss, eval_teacher_loss = [], []

    log_path = os.path.join(log_dir)
    if not os.path.exists(log_path):
        print(f"Log directory not found: {log_path}")
        return None

    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(log_path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    # Extract available tags
    available_tags = ea.Tags()["scalars"]

    # Training loss & steps
    if "train/loss" in available_tags:
        events = ea.Scalars("train/loss")
        train_loss = [e.value for e in events]
        train_steps = [e.step for e in events]
    elif "train/train_loss" in available_tags:
        events = ea.Scalars("train/train_loss")
        train_loss = [e.value for e in events]
        train_steps = [e.step for e in events]

    # Validation loss & steps
    if "eval/loss" in available_tags:
        events = ea.Scalars("eval/loss")
        val_loss = [e.value for e in events]
        val_steps = [e.step for e in events]

    # Validation accuracy
    if "eval/accuracy" in available_tags:
        events = ea.Scalars("eval/accuracy")
        val_acc = [e.value for e in events]

    # Student and teacher losses
    if "train/student_loss" in available_tags:
        events = ea.Scalars("train/student_loss")
        student_loss = [e.value for e in events]
    if "train/teacher_loss" in available_tags:
        events = ea.Scalars("train/teacher_loss")
        teacher_loss = [e.value for e in events]
    if "train/eval/student_loss" in available_tags:
        events = ea.Scalars("train/eval/student_loss")
        eval_student_loss = [e.value for e in events]
    if "train/eval/teacher_loss" in available_tags:
        events = ea.Scalars("train/eval/teacher_loss")
        eval_teacher_loss = [e.value for e in events]

    return train_steps, train_loss, val_steps, val_loss, val_acc, student_loss, teacher_loss, eval_student_loss, eval_teacher_loss

# Function to gather model logs from a given folder
def gather_model_logs(base_folder):
    model_paths = {}
    mbert_models = {}
    other_models = {}

    # Walk through the base folder to get all subdirectories (model folders)
    for root, dirs, files in os.walk(base_folder):
        for dir_name in dirs:
            log_dir = os.path.join(root, dir_name, 'logs')
            if os.path.exists(log_dir):
                if "_mbert_" in dir_name.lower():
                    mbert_models[dir_name] = log_dir
                else:
                    other_models[dir_name] = log_dir

    # First include mBERT-related models, then the others
    model_paths.update(mbert_models)
    model_paths.update(other_models)

    return model_paths


# Function to reduce data frequency
def downsample_data(steps, values, every_n=10):
    if not steps or not values:
        return [], []
    return steps[::every_n], values[::every_n]

# Function to get gray or brown color based on model name
def get_neutral_color(model_name):
    # Generate a hash based on the model name to get a consistent number
    hash_val = int(hashlib.md5(model_name.encode()).hexdigest(), 16)
    
    # Determine whether to use gray or brown (50% chance each)
    color_type = hash_val % 2  # 0-1 different options
    
    if color_type == 0:  # Gray shades
        shade = 0.4 + (hash_val % 100) / 250  # Range approximately 0.4-0.8
        return (shade, shade, shade)
    else:  # Brown tones
        r = 0.6 + (hash_val % 50) / 200  # 0.6-0.85
        g = 0.45 + (hash_val % 40) / 200  # 0.45-0.65
        b = 0.35 + (hash_val % 30) / 200  # 0.35-0.5
        return (r, g, b)
        
# Function to determine line style based on model name
def get_line_properties(model_name):
    if "student" in model_name.lower():
        gray_shade = get_neutral_color(model_name)
        return {
            "color": gray_shade,  # Model-specific gray shade
            "alpha": 0.75,        # 75% transparency
            "linewidth": 1.0      # Slightly thinner line
        }
    else:
        return {
            "color": None,        # Default color from cycle
            "alpha": 1.0,         # Full opacity
            "linewidth": 1.5      # Thicker line for emphasis
        }

# Function to generate simplified legend names
def get_simplified_legend_name(model_name, metric_type):
    # Check if it's a student or distilled model
    model_type = ""
    if "student" in model_name.lower():
        model_type = "Student"
    elif "distilled" in model_name.lower():
        model_type = "Distilled"
    
    # Check if it's mBERT or mBERT-mt
    bert_type = ""
    if "mbert-mt" in model_name.lower():
        bert_type = "mBERT-mt"
    elif "mbert-sk" in model_name.lower():
        bert_type = "mBERT-sk"
    elif "mbert-sw" in model_name.lower():
        bert_type = "mBERT-sw"
    elif "mbert" in model_name.lower():
        bert_type = "mBERT"
        
    if "svd" in model_name.lower():
        bert_type = "mBERT-svd"
        bert_type += "_" + "_".join(model_name.split("_")[4:])
    elif "truncation" in model_name.lower():
        bert_type = "mBERT-truncate"
        bert_type += "_" + "_".join(model_name.split("_")[4:])
    
    # Combine the information
    return f"{bert_type} - {model_type}"

# Gather the logs for all models in the parent folder
model_paths = gather_model_logs(base_folder)

# Extract logs for each model
logs = {name: extract_logs(log_dir) for name, log_dir in model_paths.items()}

# Create the plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Use a serif font for publication quality
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.labelsize"] = 14  # Larger labels
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["lines.markersize"] = 6

# Plot Training & Validation Loss
plt.figure(figsize=(8, 4.5))
for name, data in logs.items():
    print(name)
    if data is not None:
        train_steps, train_loss, val_steps, val_loss, _, student_loss, teacher_loss, eval_student_loss, eval_teacher_loss = data
        
        # Get line properties based on model name
        line_props = get_line_properties(name)
        
        if val_loss:
            # Get simplified legend name
            legend_name = get_simplified_legend_name(name, "Val Loss")
            
            # Downsample the data
            downsampled_steps, downsampled_loss = downsample_data(val_steps, val_loss, plot_every_n)
            plt.plot(downsampled_steps, downsampled_loss, '--', 
                    label=legend_name,
                    color=line_props["color"],
                    alpha=line_props["alpha"],
                    linewidth=line_props["linewidth"])

# plt.xlabel("Steps")
# plt.ylabel("Loss")
plt.title(f"Validation Loss for {language_name}", fontsize=14)
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.savefig(f"plots/validation_loss_{language_name.lower()}.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(8, 4.5))
for name, data in logs.items():
    if data is not None:
        _, _, val_steps, _, val_acc, _, _, _, _ = data
        
        # Get line properties based on model name
        line_props = get_line_properties(name)
        
        if val_acc:
            # Get simplified legend name
            legend_name = get_simplified_legend_name(name, "Val Acc")
            
            # Downsample the data
            downsampled_steps, downsampled_acc = downsample_data(val_steps, val_acc, plot_every_n)
            downsampled_acc = gaussian_filter1d(downsampled_acc, sigma=8)
            plt.plot(downsampled_steps, downsampled_acc, 
                    label=legend_name,
                    color=line_props["color"],
                    alpha=line_props["alpha"],
                    linewidth=line_props["linewidth"])

# plt.xlabel("Steps")
# plt.ylabel("Accuracy")
plt.title(f"Validation Accuracy for {language_name}", fontsize=14)
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.savefig(f"plots/validation_accuracy_{language_name.lower()}.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot Student vs. Teacher Loss during Training and Evaluation
plt.figure(figsize=(8, 4.5))
for name, data in logs.items():
    if data is not None:
        train_steps, _, val_steps, _, _, student_loss, _, eval_student_loss, _ = data
        
        # Get line properties based on model name
        line_props = get_line_properties(name)

        # Plot Student Loss (Evaluation)
        min_eval_length = min(len(val_steps), len(eval_student_loss))
        if min_eval_length > 0:
            # Get simplified legend name
            legend_name = get_simplified_legend_name(name, "Student Loss")
            
            val_steps_slice = val_steps[:min_eval_length]
            eval_student_loss_slice = eval_student_loss[:min_eval_length]
            # Downsample the data
            downsampled_steps, downsampled_loss = downsample_data(val_steps_slice, eval_student_loss_slice, plot_every_n)
            plt.plot(downsampled_steps, downsampled_loss, '-.', 
                    label=legend_name,
                    color=line_props["color"],
                    alpha=line_props["alpha"],
                    linewidth=line_props["linewidth"])

# plt.xlabel("Steps")
# plt.ylabel("Loss")
plt.title(f"Student Validation Loss for {language_name}", fontsize=14)
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.savefig(f"plots/student_loss_{language_name.lower()}.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 4.5))
for name, data in logs.items():
    if data is not None:
        train_steps, _, val_steps, _, _, _, teacher_loss, _, eval_teacher_loss = data
        
        # Get line properties based on model name
        line_props = get_line_properties(name)

        # Plot Teacher Loss (Evaluation)
        min_eval_length = min(len(val_steps), len(eval_teacher_loss))
        if min_eval_length > 0:
            # Get simplified legend name
            legend_name = get_simplified_legend_name(name, "Teacher Loss")
            
            val_steps_slice = val_steps[:min_eval_length]
            eval_teacher_loss_slice = eval_teacher_loss[:min_eval_length]
            # Downsample the data
            downsampled_steps, downsampled_loss = downsample_data(val_steps_slice, eval_teacher_loss_slice, plot_every_n)
            plt.plot(downsampled_steps, downsampled_loss, ':', 
                    label=legend_name,
                    color=line_props["color"],
                    alpha=line_props["alpha"],
                    linewidth=line_props["linewidth"])

# plt.xlabel("Steps")
# plt.ylabel("Loss")
plt.title(f"Teacher Validation Loss for {language_name}", fontsize=14)
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.savefig(f"plots/teacher_loss_{language_name.lower()}.png", dpi=300, bbox_inches="tight")
plt.show()