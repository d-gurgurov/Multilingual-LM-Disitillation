import pandas as pd
import torch
import os
import random
import argparse
from math import log, exp
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

def find_latest_checkpoint(model_path: str) -> str:
    """
    Finds the latest checkpoint directory within the given model folder.
    """
    # Try loading from Hugging Face if not found locally
    try:
        AutoModelForMaskedLM.from_pretrained(model_path)
        model = model_path
        return model
    except Exception as e:
        print(f"Model not found on Hugging Face!")

    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
    if not checkpoints:
        return model_path  # If no checkpoint folders, assume model is in base path
    
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    return os.path.join(model_path, checkpoints[-1])  # Return latest checkpoint path
   
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

class PseudoPerplexity:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
        if args.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        pseudo_perplexities = []
        for sentence in tqdm(sentences, desc="Computing pseudo-perplexity"):
            # Tokenize the sentence using the model's tokenizer
            tokenized_sentence = self.tokenizer.encode(
                sentence, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            num_tokens = tokenized_sentence.shape[-1]

            # Count the number of words in the sentence
            num_words = len(word_tokenize(sentence))  # Using NLTK tokenizer for word count

            # Calculate pseudo-log-likelihood and pseudo-perplexity
            pseudo_log_likelihood = self.pseudo_log_likelihood(tokenized_sentence)
            pseudo_perplexity = exp(-1 / num_tokens * pseudo_log_likelihood)

            # Normalize by word count
            word_level_normalized_ppl = pseudo_perplexity / num_words
            pseudo_perplexities.append(word_level_normalized_ppl)
        
        return {"values": pseudo_perplexities, "average": sum(pseudo_perplexities) / len(pseudo_perplexities)}

    def pseudo_log_likelihood(self, tokenized_sentence: torch.Tensor) -> float:
        pseudo_log_likelihood = 0
        for token_position, original_token_id in enumerate(tokenized_sentence.squeeze()):
            masked_sentence = tokenized_sentence.clone()
            masked_sentence[:, token_position] = self.tokenizer.mask_token_id
            with torch.no_grad():
                output = self.model(input_ids=masked_sentence)
                logits = output.logits.squeeze()
            probability = logits[token_position].softmax(dim=0)[original_token_id]
            pseudo_log_likelihood += log(probability)
        return pseudo_log_likelihood


def compute_pseudo_perplexity(model_name: str, model_path: str, language_code: str, output_file: str):
    random.seed(42)
    torch.manual_seed(42)

    dataset = load_dataset("facebook/flores", name=language_code)
    sentences = dataset["devtest"]["sentence"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = PseudoPerplexity(model_path, device)
    results = metric(sentences)

    df = pd.DataFrame({"text": sentences, "pseudo_perplexity": results["values"]})
    df.to_csv(output_file, index=False)

    print(f"[{model_name}] Pseudo-perplexity saved to {output_file}")
    return results["average"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory or checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")
    parser.add_argument("--language_code", type=str, required=True, help="Language code for the FLORES-200 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output pseudo-perplexity results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    latest_checkpoint = find_latest_checkpoint(args.model_path)
    output_file = os.path.join(args.output_dir, f"{args.model_path}_pseudo_perplexity.csv")
    avg_ppl = compute_pseudo_perplexity(args.model_path, latest_checkpoint, args.language_code, output_file)
    print(args.model_path)
    print(avg_ppl)

