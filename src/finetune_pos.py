# import dependencies
import argparse
import numpy as np
import os
import json
import random
import torch
from transformers import set_seed

from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoModel,
)
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer, Fuse
from adapters.composition import Stack
import pandas as pd
import conllu

# useful functions
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model for a POS tagging task using Universal Dependencies.")
    parser.add_argument("--language", type=str, default="", help="Language at hand")
    parser.add_argument("--output_dir", type=str, default="./training_output", help="Output directory for training results")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="Name of the pre-trained model")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-multilingual-cased", help="Name of the pre-trained tokenizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size per device during evaluation")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy during training")
    parser.add_argument("--save_strategy", type=str, default="no", help="Saving strategy during training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--ud_train_file", type=str, default="", help="Path to UD train file")
    parser.add_argument("--ud_dev_file", type=str, default="", help="Path to UD dev file")
    parser.add_argument("--ud_test_file", type=str, default="", help="Path to UD test file")
    return parser.parse_args()

args = parse_arguments()

def compute_metrics(p, label_names):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval", trust_remote_code=True)
    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if k not in flattened_results.keys():
            flattened_results[k+"_f1"] = results[k]["f1"]
    return flattened_results

def load_ud_file(file_path):
    """
    Load a Universal Dependencies file and extract tokens and POS tags.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    sentences = conllu.parse(content)
    all_tokens = []
    all_pos_tags = []
    
    for sentence in sentences:
        tokens = []
        pos_tags = []
        for token in sentence:
            # Skip multi-word tokens and empty nodes
            if isinstance(token["id"], int):
                tokens.append(token["form"])
                pos_tags.append(token["upos"])
        
        if tokens:  # Only add non-empty sentences
            all_tokens.append(tokens)
            all_pos_tags.append(pos_tags)
    
    return {"tokens": all_tokens, "pos_tags": all_pos_tags}

def prepare_dataset(train_file, dev_file, test_file):
    """
    Prepare datasets from UD files
    """
    train_data = load_ud_file(train_file)
    dev_data = load_ud_file(dev_file)
    test_data = load_ud_file(test_file)
    
    # Create datasets
    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)
    test_dataset = Dataset.from_dict(test_data)
    
    # Combine into dataset dict
    return {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}

def main():
    # set the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # Load UD datasets
    dataset = prepare_dataset(args.ud_train_file, args.ud_dev_file, args.ud_test_file)
    print(dataset)
    # Extract unique POS tags and create mapping
    all_pos_tags = set()
    for split in dataset.values():
        for tags in split["pos_tags"]:
            all_pos_tags.update(tags)
    
    label_names = sorted(list(all_pos_tags))
    id2label = {id_: label for id_, label in enumerate(label_names)}
    label2id = {label: id_ for id_, label in enumerate(label_names)}
    
    # Map string labels to IDs
    for split in dataset:
        dataset[split] = dataset[split].map(
            lambda examples: {"pos_tag_ids": [[label2id[tag] for tag in tags] for tags in examples["pos_tags"]]},
            batched=True
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], max_length=512, is_split_into_words=True)
        
        total_adjusted_labels = []
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["pos_tag_ids"][k]
            i = -1
            adjusted_label_ids = []
        
            for wid in word_ids_list:
                if wid is None:
                    adjusted_label_ids.append(-100)
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    adjusted_label_ids.append(existing_label_ids[i])
                
            total_adjusted_labels.append(adjusted_label_ids)
        
        tokenized_samples["labels"] = total_adjusted_labels
        return tokenized_samples

    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(tokenize_adjust_labels, batched=True)

    def find_latest_checkpoint(model_path: str) -> str:
        """
        Finds the latest checkpoint directory within the given model folder.
        """
        # Try loading from Hugging Face if not found locally
        print(f"Model not found locally. Attempting to load from Hugging Face: {model_path}")
        try:
            AutoAdapterModel.from_pretrained(model_path)
            model = model_path
            return model
        except Exception as e:
            print(f"Model not found in local directories or Hugging Face: {e}")

        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
        if not checkpoints:
            return model_path  # If no checkpoint folders, assume model is in base path
        
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
        return os.path.join(model_path, checkpoints[-1])  # Return latest checkpoint path
    
    # prepare model
    model_path = find_latest_checkpoint(args.model_name)
    config = AutoConfig.from_pretrained(model_path, id2label=id2label, label2id=label2id)
    model = AutoAdapterModel.from_pretrained(model_path, config=config)
    print("Original number of model parameters:", model.num_parameters())

    # Set up POS tagging adapter
    model.add_adapter("pos")
    model.add_tagging_head("pos", num_labels=len(label_names), id2label=id2label)
    model.train_adapter(["pos"])

    print(model.adapter_summary())

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        warmup_ratio=0.05,
        logging_steps=20,
    )
    
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_names)
    )

    # train model
    trainer.train()

    # test model
    test_results = trainer.evaluate(tokenized_dataset["test"])
    output_file_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(output_file_path, "w") as f:
        json.dump(test_results, f)

    # Save the label mappings for future use
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)

if __name__ == "__main__":
    main()