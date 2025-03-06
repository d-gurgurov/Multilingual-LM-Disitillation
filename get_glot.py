import os
from datasets import load_dataset

# Define the local directory to save the dataset
data_dir = 'data_slovak'

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Load the Slovak monolingual dataset
dataset = load_dataset("cis-lmu/GlotCC-v1", name="slk-Latn", split="train")

# Limit to 100,000 texts
dataset = dataset.select(range(min(100000, len(dataset))))  # Ensures we don't exceed dataset size

# Save the dataset to the local directory
dataset.save_to_disk(data_dir)

print(f"Dataset saved to {data_dir} with {len(dataset)} examples")
