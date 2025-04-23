# import dependencies
import argparse
import numpy as np
import os
import json
import random
import torch
from transformers import set_seed

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoModel,
)
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer, Fuse
from adapters.composition import Stack

# useful functions
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model for a sentiment analysis task.")
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


def main():

    # set the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], max_length=512, is_split_into_words=True)
        #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
        #so the new keys [input_ids, labels (after adjustment)]
        #can be added to the datasets dict for each train test validation split
        total_adjusted_labels = []
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []
        
            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid!=prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i])
                
            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        tokenized_samples["labels"] = [list(map(int, x)) for x in tokenized_samples["labels"]]

        return tokenized_samples

    # Define language mapping
    languages_mapping = {
        "swh_Latn": "sw", "slk_Latn": "sk", "mlt_Latn": "mt"
    } 

    # prepare data
    dataset = load_dataset("wikiann", languages_mapping[args.language])
    label_names = dataset["train"].features["ner_tags"].feature.names
    id2label = {id_: label for id_, label in enumerate(label_names)}
    label2id = {label: id_ for id_, label in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

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

    # no language adapter used
    model.add_adapter("ner")
    model.add_tagging_head("ner", num_labels=len(label_names), id2label=id2label)
    model.train_adapter(["ner"])

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

if __name__ == "__main__":
    main()