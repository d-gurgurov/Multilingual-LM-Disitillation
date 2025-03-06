import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Define model log directories

model_paths = {
    "MSE - Random": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/distilled_model_2_double_loss_alpha_0.5_mse_unsoft_random/logs",
    "MSE - Teacher": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/distilled_model_2_double_loss_alpha_0.5_mse_unsoft/logs",
    "Student (No Distil.) - Random": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/student_model_2_random/logs",
    "Student (No Distil.) - Teacher": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/student_model_2/logs",
    }

model_paths = {
    "MSE - 6": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/distilled_model_6_double_loss_alpha_0.2_mse/logs",
    "MSE - 4": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/distilled_model_4_double_loss_alpha_0.2_mse/logs",
    "MSE - 2": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/distilled_model_2_double_loss_alpha_0.2_mse_unsoft/logs",
    "Student (No Distil.) - 6": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/student_model_6/logs",
    "Student (No Distil.) - 4": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/student_model_4/logs",
    "Student (No Distil.) - 2": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/student_model_2/logs",
    "mBERT - fine-tune": "/netscratch/dgurgurov/projects2025/distillation_mbert/models/mbert_finetuned/logs",
    }

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

# Extract logs for each model
logs = {name: extract_logs(log_dir) for name, log_dir in model_paths.items()}

# Create the plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
for name, data in logs.items():
    if data is not None:
        train_steps, train_loss, val_steps, val_loss, _, student_loss, teacher_loss, eval_student_loss, eval_teacher_loss = data
        # if train_loss:
        #     plt.plot(train_steps, train_loss, label=f"Train Loss ({name})")
        if val_loss:
            plt.plot(val_steps, val_loss, '--', label=f"Val Loss ({name})")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.xscale("log")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()
plt.savefig("plots/training_validation_loss.png", dpi=300)
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(10, 5))
for name, data in logs.items():
    if data is not None:
        _, _, val_steps, _, val_acc, _, _, _, _ = data
        if val_acc:
            plt.plot(val_steps, val_acc, label=f"Val Acc ({name})")

plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.yscale('log')
plt.xscale("log")
plt.legend()
plt.grid()
plt.savefig("plots/validation_accuracy.png", dpi=300)
plt.show()

# Plot Student vs. Teacher Loss during Training and Evaluation
plt.figure(figsize=(10, 5))
for name, data in logs.items():
    if data is not None:
        train_steps, _, val_steps, _, _, student_loss, _, eval_student_loss, _ = data

        # Plot Student Loss (Training)
        min_train_length = min(len(train_steps), len(student_loss))
        if min_train_length > 0:
            plt.plot(train_steps[:min_train_length], student_loss[:min_train_length], label=f"Train Student Loss ({name})")

        # Plot Student Loss (Evaluation)
        min_eval_length = min(len(val_steps), len(eval_student_loss))
        if min_eval_length > 0:
            plt.plot(val_steps[:min_eval_length], eval_student_loss[:min_eval_length], '-.', label=f"Eval Student Loss ({name})")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Student Loss during Training and Evaluation")
plt.legend()
plt.grid()
plt.savefig("plots/student_loss.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
for name, data in logs.items():
    if data is not None:
        train_steps, _, val_steps, _, _, _, teacher_loss, _, eval_teacher_loss = data

        # Plot Teacher Loss (Training)
        min_train_length = min(len(train_steps), len(teacher_loss))
        if min_train_length > 0:
            plt.plot(train_steps[:min_train_length], teacher_loss[:min_train_length], '--', label=f"Train Teacher Loss ({name})")

        # Plot Teacher Loss (Evaluation)
        min_eval_length = min(len(val_steps), len(eval_teacher_loss))
        if min_eval_length > 0:
            plt.plot(val_steps[:min_eval_length], eval_teacher_loss[:min_eval_length], ':', label=f"Eval Teacher Loss ({name})")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Teacher Loss during Training and Evaluation")
plt.legend()
plt.grid()
plt.savefig("plots/teacher_loss.png", dpi=300)
plt.show()

