import os
import json
import time
import argparse
from torch import nn
from transformers import (
    BertTokenizer, BertTokenizerFast, BertForMaskedLM,
    XLMRobertaTokenizer, XLMRobertaTokenizerFast, XLMRobertaForMaskedLM,
    PreTrainedModel
)
from datasets import load_from_disk

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reduce vocabulary size of a BERT or XLM-R model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the pretrained model')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Name or path of the tokenizer')
    parser.add_argument('--model_type', type=str, choices=['bert', 'xlm-roberta'], required=True, help='Model type')
    parser.add_argument('--output_name', type=str, required=True, help='Name for the output model directory')
    parser.add_argument('--threshold_percent', type=float, default=0.005, help='Threshold percentage for token selection')
    parser.add_argument('--target_vocab_size', type=int, default=None, help='Target vocabulary size (overrides threshold_percent)')
    parser.add_argument('--output_dir', type=str, default='reduce_vocab', help='Directory to save the output model')
    parser.add_argument('--tokens_dir', type=str, default='reduce_vocab', help='Directory to save token frequencies')
    return parser.parse_args()

class ReducedVocabModel(PreTrainedModel):
    def __init__(self, model, new_vocab, model_type, model_name='new_model'):
        super().__init__(model.config)
        self.model = model
        self.new_vocab = new_vocab
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "bert":
            self.tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
        elif model_type == "xlm-roberta":
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.tokenizer_name)
        else:
            raise ValueError("Unsupported model type")

        # self.tokenizer = self._update_tokenizer()
        self._update_embeddings()
        self._save_model()

    def _update_tokenizer(self):
        if self.model_type == "bert":
            vocab_path = os.path.join(self.model_name, 'vocab.txt')
            with open(vocab_path, 'w') as fw:
                fw.write('\n'.join(self.new_vocab))
            tokenizer = BertTokenizerFast(vocab_file=vocab_path, do_lower_case=False)

            tokenizer_config_path = os.path.join(self.model_name, 'tokenizer_config.json')
            with open(tokenizer_config_path, 'w') as fw:
                json.dump({"do_lower_case": False, "model_max_length": 512}, fw)
        
        elif self.model_type == "xlm-roberta":
            pass

        return tokenizer

    def _update_embeddings(self):
        old_embeddings = self.model.get_input_embeddings()
        new_num_tokens = len(self.new_vocab)
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)

        if self.model_type == "bert":
            vocab = list(self.tokenizer.get_vocab().keys())
        elif self.model_type == "xlm-roberta":
            vocab = self.tokenizer.get_vocab()
            print(len(vocab))

        for i, token in enumerate(self.new_vocab):
            if token in vocab:
                idx = list(vocab.keys()).index(token)
                new_embeddings.weight.data[i] = old_embeddings.weight.data[idx]

        self.model.set_input_embeddings(new_embeddings)
        self.model.config.vocab_size = new_num_tokens
        self.model.tie_weights()
        self.vocab = self.new_vocab

    def _save_model(self):
        os.makedirs(self.model_name, exist_ok=True)
        self.model.save_pretrained(self.model_name)
        self.tokenizer.save_pretrained(self.model_name)
        with open(os.path.join(self.model_name, 'vocab.txt'), 'w') as fw:
            fw.write('\n'.join(self.vocab))
        print(f"Model saved to {self.model_name}, num tokens: {len(self.vocab)}")

def main():
    global args, bert_vocab
    args = parse_arguments()

    os.makedirs(args.tokens_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print("Data loaded!")

    if args.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        model = BertForMaskedLM.from_pretrained(args.model_name, attn_implementation="eager")
        essential_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    elif args.model_type == "xlm-roberta":
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = XLMRobertaForMaskedLM.from_pretrained(args.model_name)
        essential_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<mask>']
    else:
        raise ValueError("Unsupported model type")

    bert_vocab = list(tokenizer.get_vocab().keys())
    print("Original vocab size:", len(bert_vocab))
    print("Original number of model parameters:", model.num_parameters())

    dataset_name = os.path.basename(args.dataset_path)
    freq_file = os.path.join(args.tokens_dir, f'{dataset_name}_freqs.json')

    if os.path.exists(freq_file):
        print(f"Frequency file already exists at {freq_file}. Loading frequencies...")
        with open(freq_file, 'r') as infile:
            lang_tokens = json.load(infile)
        lang_tokens_unique = {token: 1 for token in lang_tokens}  # Dummy unique freq just to proceed
        print(f"Loaded {len(lang_tokens)} token frequencies.")
    else:
        lang_tokens = {}
        lang_tokens_unique = {}
        print("Calculating token frequencies...")
        start_time = time.time()
        for sample in dataset["content"]:
            tokens = tokenizer.tokenize(sample)
            for token in tokens:
                lang_tokens[token] = lang_tokens.get(token, 0) + 1
            for token in set(tokens):
                lang_tokens_unique[token] = lang_tokens_unique.get(token, 0) + 1
        print("Time taken for frequency calculation:", time.time() - start_time)

        with open(freq_file, 'w') as outfile:
            json.dump(lang_tokens, outfile)
        print(f"Token frequencies saved to {freq_file}")

    if args.target_vocab_size is not None:
        print(f"Using target vocabulary size: {args.target_vocab_size}")
        sorted_tokens = sorted(lang_tokens_unique.items(), key=lambda x: x[1], reverse=True)
        target_size = args.target_vocab_size + len(essential_tokens)
        target_size = min(target_size, len(sorted_tokens))
        selected_tokens = [tok for tok, _ in sorted_tokens[:target_size]]
    else:
        print(f"Using threshold percentage: {args.threshold_percent}%")
        thresh = int(len(dataset["content"]) * args.threshold_percent / 100)
        print(f"Minimum document frequency threshold: {thresh}")
        selected_tokens = [tok for tok, count in lang_tokens_unique.items() if count >= thresh]

    selected_tokens = list(set(selected_tokens + essential_tokens))
    print("New vocab size:", len(selected_tokens))

    print(f"Vocabulary reduction: {len(bert_vocab)} → {len(selected_tokens)} tokens")
    print(f"Reduction ratio: {len(selected_tokens)/len(bert_vocab):.2%}")

    print("Creating reduced vocabulary model...")
    start_time = time.time()
    reduced_model = ReducedVocabModel(model, selected_tokens, args.model_type, output_path)
    print("Reduced number of model parameters:", reduced_model.model.num_parameters())
    print("Parameter reduction ratio:", f"{reduced_model.model.num_parameters()/model.num_parameters():.2%}")
    print("Time taken for model reduction:", time.time() - start_time)

if __name__ == "__main__":
    main()
