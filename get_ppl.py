import pandas as pd
import torch
import os
import random
from math import log, exp
from abc import ABC
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel
from tensorboard.backend.event_processing import event_accumulator


# Define the models and their paths
model_paths = {
    "Distilled (Double Loss)": "./models/distilled_model_4_double_loss",
    "Distilled (Triple Loss)": "./models/distilled_model_4_triple_loss",
    "Distilled (Double Loss + D-BERT)": "./models/distilled_model_4_double_loss_dbert",
    "Distilled (Double Loss + D-BERT Random)": "./models/distilled_model_4_double_loss_dbert_random",
    "mBERT Fine-Tuned": "./models/mbert_finetuned",
    "Student Model (No Distillation)": "./models/student_model_4"
}


class Metric(ABC):
    def __init__(self, model: PreTrainedModel, device: torch.device) -> None:
        self.device = device  # Store the device (CPU/GPU)

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        raise NotImplementedError


class PseudoPerplexity(Metric):
    """
    Computes pseudo-perplexity for a list of sentences using a Masked Language Model (MLM).
    """

    def __init__(self, model: PreTrainedModel, device: torch.device):
        super().__init__(model, device)
        self.model: PreTrainedModel = model.to(self.device)  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        assert len(sentences) > 0

        pseudo_perplexities: list[float] = []
        for sentence in tqdm(sentences, desc="Computing pseudo-perplexity"):
            tokenized_sentence: torch.Tensor = self.tokenizer.encode(
                sentence, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)  # type: ignore
            num_tokens = tokenized_sentence.shape[-1]

            pseudo_log_likelihood = self.pseudo_log_likelihood(tokenized_sentence)
            pseudo_perplexity = exp(-1 / num_tokens * pseudo_log_likelihood)
            pseudo_perplexities.append(pseudo_perplexity)

        average_pseudo_perplexity: float = sum(pseudo_perplexities) / len(pseudo_perplexities)
        return {"values": pseudo_perplexities, "average": average_pseudo_perplexity}

    def pseudo_log_likelihood(self, tokenized_sentence: torch.Tensor) -> float:
        pseudo_log_likelihood = 0
        for token_position, original_token_id in enumerate(tokenized_sentence.squeeze()):
            masked_sentence = tokenized_sentence.clone().to(self.device)
            masked_sentence[:, token_position] = self.tokenizer.mask_token_id  # type: ignore
            with torch.no_grad():
                output = self.model(input_ids=masked_sentence)
                logits: torch.Tensor = output.logits.squeeze()
            probabilities = logits[token_position].softmax(dim=0)
            probability = probabilities[original_token_id]
            pseudo_log_likelihood += log(probability)

        return pseudo_log_likelihood


def compute_pseudo_perplexity(model_name: str, model_path: str, language_code: str, output_file: str, seed: int = 42):
    """
    Computes pseudo-perplexity for a subset of sentences from the FLORES-200 dataset.

    :param model_name: The name of the model.
    :param model_path: The path to the model checkpoint.
    :param language_code: The language code for the FLORES-200 dataset.
    :param output_file: The path to save the output CSV file with pseudo-perplexities.
    :param seed: Random seed for reproducibility.
    :returns: The average pseudo-perplexity.
    """
    # Set the random seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Load FLORES-200 for Maltese (mlt_Latn)
    dataset = load_dataset("facebook/flores", name=language_code)
    sentences = dataset["devtest"]["sentence"]  # type: ignore

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    pseudo_perplexity_metric = PseudoPerplexity(model, device)

    # Compute pseudo-perplexity
    results = pseudo_perplexity_metric(sentences)

    # Save results
    df = pd.DataFrame({"text": sentences, "pseudo_perplexity": results["values"]})
    df.to_csv(output_file, index=False)

    print(f"[{model_name}] Pseudo-perplexity saved to {output_file}")
    print(f"[{model_name}] Average Pseudo-Perplexity: {results['average']}")

    return results["average"]  # type: ignore

def find_latest_checkpoint(base_path: str) -> str:
    """
    Finds the latest checkpoint directory for the given language.
    """
    lang_path = base_path
    checkpoints = [d for d in os.listdir(lang_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    return os.path.join(lang_path, latest_checkpoint)

if __name__ == "__main__":
    language_code = "mlt_Latn"  # Maltese in FLORES-200
    output_dir = "pseudo_perplexity_results"
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for model_name, model_path in model_paths.items():
        output_file = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_pseudo_perplexity.csv")
        avg_ppl = compute_pseudo_perplexity(model_name, find_latest_checkpoint(model_path), language_code, output_file)
        results[model_name] = avg_ppl

    # Save summary results
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Average Pseudo-Perplexity"])
    results_df.to_csv(os.path.join(output_dir, "summary_pseudo_perplexity.csv"))
    print("\nAll results saved in:", output_dir)
