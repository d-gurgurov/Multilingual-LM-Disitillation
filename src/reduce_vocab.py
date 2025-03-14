import os
import json
import time
import argparse
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast, PreTrainedModel
from datasets import load_from_disk

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reduce vocabulary size of a BERT model')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name or path of the pretrained model')
    parser.add_argument('--tokenizer_name', type=str, required=True,
                        help='Name or path of the tokenizer')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Name for the output model directory')
    parser.add_argument('--threshold_percent', type=float, default=0.005,
                        help='Threshold percentage for token selection (default: 0.005)')
    parser.add_argument('--output_dir', type=str, default='reduce_vocab',
                        help='Directory to save the output model (default: reduce_vocab)')
    parser.add_argument('--tokens_dir', type=str, default='tokens_freqs',
                        help='Directory to save token frequencies (default: tokens_freqs)')
    return parser.parse_args()

class ReducedVocabBert(PreTrainedModel):
    def __init__(self, model, new_vocab, model_name='new_model'):
        super().__init__(model.config)
        self.model = model
        self.new_vocab = new_vocab
        self.model_name = model_name

        self.tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)

        self.tokenizer = self._update_tokenizer()
        self._update_embeddings()
        self._save_model()
        
    def _update_tokenizer(self):
        """
        Creates a new tokenizer with the reduced vocabulary and saves it properly.
        """
        vocab_dict = {token: idx for idx, token in enumerate(self.new_vocab)}

        vocab_path = os.path.join(self.model_name, 'vocab.txt')
        with open(vocab_path, 'w') as fw:
            fw.write('\n'.join(self.new_vocab))

        tokenizer = BertTokenizerFast(vocab_file=vocab_path, do_lower_case=False)

        tokenizer_config_path = os.path.join(self.model_name, 'tokenizer_config.json')
        with open(tokenizer_config_path, 'w') as fw:
            json.dump({"do_lower_case": False, "model_max_length": 512}, fw)

        return tokenizer

    def _update_embeddings(self):
        old_embeddings = self.model.get_input_embeddings()
        new_num_tokens = len(self.new_vocab)
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)

        vocab = []
        for i, token in enumerate(self.new_vocab):
            if token in bert_vocab:
                idx = bert_vocab.index(token)
                new_embeddings.weight.data[i] = old_embeddings.weight.data[idx]
            vocab.append(token)

        self.model.set_input_embeddings(new_embeddings)
        self.model.config.vocab_size = new_num_tokens
        self.model.tie_weights()
        self.vocab = vocab

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
    
    # Create output directory structure
    os.makedirs(args.tokens_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print("Data loaded!")

    print(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
    
    print(f"Loading model from {args.model_name}")
    model = BertForMaskedLM.from_pretrained(args.model_name)

    # Extract vocabulary
    bert_vocab = list(tokenizer.vocab.keys())
    print("Original vocab size:", len(bert_vocab))
    print("Original number of model parameters:", model.num_parameters())

    # Token frequency calculation
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

    # Save token frequencies
    dataset_name = os.path.basename(args.dataset_path)
    freq_file = os.path.join(args.tokens_dir, f'{dataset_name}_freqs.json')
    with open(freq_file, 'w') as outfile:
        json.dump(lang_tokens, outfile)
    print(f"Token frequencies saved to {freq_file}")

    # Select most frequent tokens based on threshold
    thresh = int(len(dataset["content"]) * args.threshold_percent / 100)
    selected_tokens = [tok for tok, count in lang_tokens_unique.items() if count >= thresh]

    # Ensure essential tokens remain
    TOKENS_TO_KEEP = ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']
    selected_tokens = list(set(selected_tokens + TOKENS_TO_KEEP))
    print("New vocab size:", len(selected_tokens))

    print("Creating reduced vocabulary model...")
    start_time = time.time()
    reduced_model = ReducedVocabBert(model, selected_tokens, output_path)
    print("Reduced number of model parameters:", reduced_model.model.num_parameters())
    print("Time taken for model reduction:", time.time() - start_time)

if __name__ == "__main__":
    main()