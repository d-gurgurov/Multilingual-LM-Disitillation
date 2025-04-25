import os
import json
import numpy as np
import csv

# Define the base directory where all models are stored
base_dir = "/netscratch/dgurgurov/projects2025/distillation_mbert/pos/slk_Latn"

# Dictionary to store results
model_f1_scores = {}

# Loop through all models in the base directory
for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    
    # Ensure it's a directory
    if not os.path.isdir(model_path):
        continue
    
    f1_scores = []
    
    # Loop through the seed folders (1, 2, 3)
    for seed in ["1", "2", "3"]:
        seed_path = os.path.join(model_path, seed, "test_metrics.json")
        
        if os.path.exists(seed_path):
            with open(seed_path, "r") as f:
                data = json.load(f)
                if "eval_overall_f1" in data:
                    f1_scores.append(data["eval_overall_f1"])
    
    # Compute and store the statistics if at least one seed has a value
    if f1_scores:
        model_f1_scores[model_name] = {
            "seed_scores": f1_scores,
            "average_f1": np.mean(f1_scores),
            "std_dev_f1": np.std(f1_scores, ddof=1)  # Use ddof=1 for sample std deviation
        }

# Print results
for model, scores in model_f1_scores.items():
    print(f"Model: {model}")
    # print(f"  Seed Scores: {scores['seed_scores']}")
    print(f"  Average F1: {scores['average_f1']:.4f}")
    print(f"  Std Dev F1: {scores['std_dev_f1']:.4f}")
    print("-" * 40)

# Define output CSV file path
csv_file_path = os.path.join(base_dir, "ner_f1_scores.csv")

# Write results to CSV file
with open(csv_file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Model Name", "Average F1 Score", "Std Dev F1 Score"])  # Header row
    
    for model, scores in model_f1_scores.items():
        writer.writerow([model, f"{scores['average_f1']:.4f}", f"{scores['std_dev_f1']:.4f}"])

print(f"Results saved to {csv_file_path}")