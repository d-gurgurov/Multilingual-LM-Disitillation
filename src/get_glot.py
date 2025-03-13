import os
from datasets import load_dataset

# Define the local directory to save the dataset
data_dir = 'data'

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Load the Maltese monolingual dataset
dataset = load_dataset("cis-lmu/GlotCC-v1", name="mlt-Latn", split="train")

# Save the dataset to the local directory
dataset.save_to_disk(data_dir)

print(f"Dataset saved to {data_dir}")
