import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import os
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from adapters import AutoAdapterModel, AdapterConfig
import torch.nn.functional as F


class LogitLensAnalyzer:
    """
    A class to apply Logit Lens analysis to multilingual transformer models.
    Works with mBERT, XLM-R, and their distilled versions.
    """
    def __init__(self, model_name, tokenizer_name=None):
        """
        Initialize the analyzer with model and tokenizer.
        
        Args:
            model_name (str): HuggingFace model name or path
            tokenizer_name (str, optional): HuggingFace tokenizer name or path.
                                           If None, uses model_name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        # Load model - try as masked LM first (most common for mBERT/XLM-R)
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.is_mlm = True
            print(f"Loaded {model_name} as MaskedLM model")
        except:
            # Fallback to base model
            self.model = AutoModel.from_pretrained(model_name, from_safetensors=True)
            self.is_mlm = False
            print(f"Loaded {model_name} as base model")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Detect if it's a distilled model
        self.is_distilled = any(name in model_name.lower() for name in ["distil", "tiny", "mini", "small"])
        if self.is_distilled:
            print("Detected distilled model variant")
        
        # Get number of layers - improved architecture detection
        self.num_layers = self._detect_num_layers()
        print(f"Model has {self.num_layers} layers")

    def _detect_num_layers(self):
        """
        Detect the number of transformer layers in the model.
        Returns the number of layers.
        """
        # Check various model architectures
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "encoder") and hasattr(self.model.bert.encoder, "layer"):
            # Standard BERT/mBERT architecture
            return len(self.model.bert.encoder.layer)
        elif hasattr(self.model, "roberta") and hasattr(self.model.roberta, "encoder") and hasattr(self.model.roberta.encoder, "layer"):
            # RoBERTa/XLM-R architecture
            return len(self.model.roberta.encoder.layer)
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            # Some encoder-only models
            return len(self.model.encoder.layer)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layer"):
            # DistilBERT style
            return len(self.model.transformer.layer)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT style
            return len(self.model.transformer.h)
        elif hasattr(self.model, "layers"):
            # Some custom architectures
            return len(self.model.layers)
        else:
            # If we can't detect, use a default test approach:
            # Run a test forward pass and check hidden states
            with torch.no_grad():
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
                outputs = self.model(input_ids, output_hidden_states=True)
                
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    # Not counting embeddings
                    return len(outputs.hidden_states) - 1
                
            # Last resort: check for modules named 'layer_X'
            count = 0
            for name, _ in self.model.named_modules():
                if "layer" in name and name.split(".")[-2] == "layer":
                    count += 1
            
            if count > 0:
                return count // 2  # Rough approximation for many architectures
                
            raise ValueError("Could not determine model architecture. Please specify num_layers manually.")

    def get_hidden_states(self, text, langs=None):
        """
        Get hidden states from all layers for the provided text.
        
        Args:
            text (str or list): Input text or list of texts
            langs (str or list, optional): Language code(s) for tokenizer
            
        Returns:
            dict: Contains tokens, input_ids, and hidden states from each layer
        """
        # Convert single text to list
        if isinstance(text, str):
            text = [text]
        
        # Handle language codes
        if langs is not None:
            if isinstance(langs, str):
                langs = [langs] * len(text)
            assert len(langs) == len(text), "Number of language codes must match number of texts"
            
            # Apply language codes if tokenizer supports it
            if hasattr(self.tokenizer, "set_lang"):
                encoding_kwargs = {"text": text}
                for i, lang in enumerate(langs):
                    self.tokenizer.set_lang(lang)
                    if i == 0:  # Just using first language for now
                        break
            else:
                # For tokenizers like XLM-R that don't need explicit language setting
                encoding_kwargs = {"text": text}
        else:
            encoding_kwargs = {"text": text}
        
        # Tokenize
        inputs = self.tokenizer(
            **encoding_kwargs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get tokens for reference
        tokens = []
        for ids in inputs["input_ids"]:
            tokens.append(self.tokenizer.convert_ids_to_tokens(ids))
        
        # Extract hidden states from each layer
        hidden_states = {}
        
        with torch.no_grad():
            # Set output_hidden_states=True to get all hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get all hidden states (including embeddings)
            all_hidden_states = outputs.hidden_states
            
            # Store each layer's hidden states
            for layer_idx in range(len(all_hidden_states)):
                hidden_states[f"layer_{layer_idx}"] = all_hidden_states[layer_idx].detach()
        
        return {
            "tokens": tokens,
            "input_ids": inputs["input_ids"],
            "hidden_states": hidden_states
        }
    
    def project_to_vocab(self, hidden_state):
        """
        Project hidden state to vocabulary space
        
        Args:
            hidden_state: Tensor of shape (batch, seq_len, hidden_dim)
            
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size)
        """
        if self.is_mlm:
            print(self.model.cls)
            # For MLM models, use the existing projection
            if hasattr(self.model, "cls"):
                # mBERT style
                print("------------------using cls")
                return self.model.cls.predictions.decoder(hidden_state)
            # Handle standard XLM-R models
            elif hasattr(self.model, "lm_head"):
                print("------------------using lm_head")
                return self.model.lm_head(hidden_state)
            # Handle XLM-R Adapter Models
            elif hasattr(self.model, "heads") and "default" in self.model.heads:
                print("------------------using heads")
                return self.model.heads["default"](hidden_state)
            elif hasattr(self.model, "bert") and hasattr(self.model.bert, "embeddings"):
                # Handle case where model has bert.embeddings
                # Get the word embeddings weight and use it for projection
                if hasattr(self.model, "cls") and hasattr(self.model.cls, "predictions") and hasattr(self.model.cls.predictions, "decoder"):
                    print("------------------using bert - cls")
                    return self.model.cls.predictions.decoder(hidden_state)
                else:
                    # Try to use embedding weights directly
                    embedding_weight = self.model.bert.embeddings.word_embeddings.weight
                    print("------------------using embedding weights")
                    return torch.matmul(hidden_state, embedding_weight.transpose(0, 1))
            else:
                raise ValueError("Unsupported MLM model architecture for projection")
        else:
            # For base models without LM head, we can't do direct projection
            # This is a placeholder - in a real application you might want to 
            # create a projection matrix or attach a classifier
            raise ValueError("Base model without LM head - can't project to vocabulary")
    
    def get_top_predictions(self, hidden_state, k=5):
        """
        Get top k predictions from a hidden state
        
        Args:
            hidden_state: Tensor of shape (batch, seq_len, hidden_dim)
            k: Number of top predictions to return
            
        Returns:
            dict with top token ids, tokens and probabilities
        """
        logits = self.project_to_vocab(hidden_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
        
        # Convert to lists
        top_indices_list = top_indices.detach().cpu().numpy()
        top_probs_list = top_probs.detach().cpu().numpy()
        
        # Convert indices to tokens
        top_tokens = []
        for batch_idx in range(top_indices_list.shape[0]):
            batch_tokens = []
            for seq_idx in range(top_indices_list.shape[1]):
                tokens = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] 
                          for idx in top_indices[batch_idx, seq_idx]]
                batch_tokens.append(tokens)
            top_tokens.append(batch_tokens)
        
        return {
            "top_indices": top_indices_list,
            "top_tokens": top_tokens,
            "top_probs": top_probs_list
        }
    
    def analyze_text(self, text, langs=None, target_layers=None, k=5):
        """
        Perform logit lens analysis on input text
        
        Args:
            text (str or list): Input text(s)
            langs (str or list, optional): Language code(s)
            target_layers (list, optional): Which layers to analyze
            k (int): Number of top predictions to return
            
        Returns:
            dict: Analysis results
        """
        # Get all hidden states
        results = self.get_hidden_states(text, langs)
        
        # Define which layers to analyze
        if target_layers is None:
            # Default: first, middle, last layers
            target_layers = [0, self.num_layers // 2, self.num_layers]
            target_layers = list(range(self.num_layers + 1))
            
        # Analyze specified layers
        layer_predictions = {}
        
        for layer_idx in target_layers:
            layer_name = f"layer_{layer_idx}"
            if layer_name in results["hidden_states"]:
                try:
                    layer_preds = self.get_top_predictions(
                        results["hidden_states"][layer_name], k=k
                    )
                    layer_predictions[layer_name] = layer_preds
                except ValueError as e:
                    print(f"Skipping {layer_name}: {e}")
        
        # Add predictions to results
        results["layer_predictions"] = layer_predictions
        
        return results

    def visualize_layer_predictions(self, results, text_idx=0, token_idx=-4):
        """
        Visualize how predictions change across layers
        
        Args:
            results: Results from analyze_text
            text_idx: Which text to visualize if multiple were provided
            token_idx: Which token position to visualize (defaults to middle token)
        """
        tokens = results["tokens"][text_idx]
        
        # Default to middle token if not specified
        if token_idx is None:
            token_idx = len(tokens) // 2
        
        # Extract the original token
        target_token = tokens[token_idx]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        layers = sorted([int(l.split('_')[1]) for l in results["layer_predictions"].keys()])
        
        # Track top tokens and their probabilities across layers
        token_to_probs = {}
        
        for layer_idx in layers:
            layer_name = f"layer_{layer_idx}"
            if layer_name not in results["layer_predictions"]:
                continue
                
            layer_preds = results["layer_predictions"][layer_name]
            top_tokens = layer_preds["top_tokens"][text_idx][token_idx]
            top_probs = layer_preds["top_probs"][text_idx][token_idx]
            
            # Record probabilities for each token
            for token, prob in zip(top_tokens, top_probs):
                if token not in token_to_probs:
                    token_to_probs[token] = [0] * len(layers)
                token_to_probs[token][layers.index(layer_idx)] = prob
        
        # Plot lines for top 5 tokens across all layers
        top_tokens_overall = sorted(
            token_to_probs.keys(), 
            key=lambda t: max(token_to_probs[t]), 
            reverse=True
        )[:10]
        
        for token in top_tokens_overall:
            plt.plot(layers, token_to_probs[token], marker='o', label=token)
        
        plt.title(f"Token predictions across layers for '{target_token}'")
        plt.xlabel("Layer")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def visualize_final_layer_predictions(self, results, text_idx=0, top_k=10):
        """
        Visualize the final layer's predictions for all tokens in the text
        
        Args:
            results: Results from analyze_text
            text_idx: Which text to visualize if multiple were provided
            top_k: Number of top predictions to show
            
        Returns:
            matplotlib figure
        """
        tokens = results["tokens"][text_idx]
        
        # Get the final layer
        layers = sorted([int(l.split('_')[1]) for l in results["layer_predictions"].keys()])
        final_layer = max(layers)
        final_layer_name = f"layer_{final_layer}"
        
        # Skip special tokens
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                           self.tokenizer.pad_token, self.tokenizer.mask_token,
                           '<s>', '</s>', '<pad>', '<mask>', '[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                valid_tokens.append(token)
                valid_indices.append(i)
        
        if not valid_tokens:
            valid_tokens = tokens
            valid_indices = list(range(len(tokens)))
        
        # Create a figure with subplots for each token
        n_tokens = len(valid_indices)
        n_cols = min(3, n_tokens)
        n_rows = (n_tokens + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot bar charts for each token
        for i, (token_idx, ax) in enumerate(zip(valid_indices, axes)):
            original_token = tokens[token_idx]
            
            if final_layer_name in results["layer_predictions"]:
                try:
                    # Get predictions for this token at the final layer
                    layer_preds = results["layer_predictions"][final_layer_name]
                    top_tokens = layer_preds["top_tokens"][text_idx][token_idx][:top_k]
                    top_probs = layer_preds["top_probs"][text_idx][token_idx][:top_k]
                    
                    # Create horizontal bar chart
                    y_pos = np.arange(len(top_tokens))
                    ax.barh(y_pos, top_probs, align='center')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_tokens)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Probability')
                    ax.set_title(f"Token: '{original_token}'")
                    
                    # Highlight the original token if it's in the top predictions
                    if original_token in top_tokens:
                        orig_idx = top_tokens.index(original_token)
                        ax.get_children()[orig_idx].set_color('orange')
                        
                except (KeyError, IndexError) as e:
                    ax.text(0.5, 0.5, f"No predictions available\n{str(e)}", 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No final layer predictions", 
                        ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Final Layer (Layer {final_layer}) Predictions", y=1.02, fontsize=16)
        return fig
    
    def visualize_final_layer_heatmap(self, results, text_idx=0, top_k=5):
        """
        Create a heatmap visualization of the final layer's predictions
        
        Args:
            results: Results from analyze_text
            text_idx: Which text to visualize if multiple were provided
            top_k: Number of top predictions to show
            
        Returns:
            matplotlib figure
        """
        tokens = results["tokens"][text_idx]
        
        # Get the final layer
        layers = sorted([int(l.split('_')[1]) for l in results["layer_predictions"].keys()])
        final_layer = max(layers)
        final_layer_name = f"layer_{final_layer}"
        
        # Skip special tokens
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                           self.tokenizer.pad_token, self.tokenizer.mask_token,
                           '<s>', '</s>', '<pad>', '<mask>', '[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                valid_tokens.append(token)
                valid_indices.append(i)
        
        if not valid_tokens:
            valid_tokens = tokens
            valid_indices = list(range(len(tokens)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, top_k * 0.8), max(8, len(valid_tokens) * 0.5)))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((len(valid_indices), top_k))
        heatmap_text = np.empty((len(valid_indices), top_k), dtype=object)
        
        for i, token_idx in enumerate(valid_indices):
            original_token = tokens[token_idx]
            
            if final_layer_name in results["layer_predictions"]:
                try:
                    # Get predictions for this token at the final layer
                    layer_preds = results["layer_predictions"][final_layer_name]
                    top_tokens = layer_preds["top_tokens"][text_idx][token_idx][:top_k]
                    top_probs = layer_preds["top_probs"][text_idx][token_idx][:top_k]
                    
                    # Fill data for heatmap
                    for j, (token, prob) in enumerate(zip(top_tokens, top_probs)):
                        heatmap_data[i, j] = prob
                        heatmap_text[i, j] = token
                        
                except (KeyError, IndexError):
                    heatmap_text[i, :] = "N/A"
            else:
                heatmap_text[i, :] = "N/A"
        
        # Create custom colormap that starts from white
        cmap = LinearSegmentedColormap.from_list('BlueGradient', ['white', 'darkblue'])
        
        # Draw heatmap with text annotations
        sns.heatmap(heatmap_data, annot=heatmap_text, fmt="", cmap=cmap, 
                   xticklabels=[f"Top {j+1}" for j in range(top_k)],
                   yticklabels=valid_tokens, ax=ax, cbar_kws={'label': 'Probability'})
        
        ax.set_title(f"Final Layer (Layer {final_layer}) Top {top_k} Predictions")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Original Token")
        
        plt.tight_layout()
        return fig
        
    def visualize_all_predictions(self, results, text_idx=0):
        """
        Create a comprehensive visualization of all predictions
        
        Args:
            results: Results from analyze_text
            text_idx: Which text to visualize if multiple were provided
            
        Returns:
            dict of matplotlib figures
        """
        figures = {}
        
        # 1. Layer-wise predictions
        figures['layer_predictions'] = self.visualize_layer_predictions(results, text_idx)
        
        # 2. Predictions heatmap
        figures['predictions_heatmap'] = self.visualize_predictions_heatmap(results, text_idx)
        
        # 3. Top predictions heatmap
        figures['top_predictions_heatmap'] = self.visualize_top_predictions_heatmap(results, text_idx)
        
        # 4. Final layer predictions
        figures['final_layer_predictions'] = self.visualize_final_layer_predictions(results, text_idx)
        
        # 5. Final layer heatmap
        figures['final_layer_heatmap'] = self.visualize_final_layer_heatmap(results, text_idx)
        
        # 6. Representation shift
        figures['representation_shift'] = self.visualize_representation_shift(results)
        
        # 7. Entropy/perplexity (if available)
        try:
            figures['perplexity'] = self.visualize_perplexity_per_layer(results)
        except Exception as e:
            print(f"Could not generate perplexity visualization: {e}")
        
        return figures

    def visualize_predictions_heatmap(self, results, text_idx=0, k=5):
        """
        Create a heatmap visualization of token predictions across layers
        
        Args:
            results: Results from analyze_text
            text_idx: Which text to visualize if multiple were provided
            k: Number of top predictions to track per token
            
        Returns:
            matplotlib figure
        """
        tokens = results["tokens"][text_idx]
        layers = sorted([int(l.split('_')[1]) for l in results["layer_predictions"].keys()])
        
        # Skip special tokens
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                           self.tokenizer.pad_token, self.tokenizer.mask_token,
                           '<s>', '</s>', '<pad>', '<mask>', '[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                valid_tokens.append(token)
                valid_indices.append(i)
        
        if not valid_tokens:
            valid_tokens = tokens
            valid_indices = list(range(len(tokens)))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.8), max(8, len(valid_tokens) * 0.5)))
        
        # Prepare data for heatmap - track probabilities of correct tokens
        heatmap_data = np.zeros((len(valid_indices), len(layers)))
        
        for i, token_idx in enumerate(valid_indices):
            original_token = tokens[token_idx]
            
            for j, layer_idx in enumerate(layers):
                layer_name = f"layer_{layer_idx}"
                
                if layer_name in results["layer_predictions"]:
                    try:
                        # Get predictions for this token at this layer
                        layer_preds = results["layer_predictions"][layer_name]
                        top_tokens = layer_preds["top_tokens"][text_idx][token_idx]
                        top_probs = layer_preds["top_probs"][text_idx][token_idx]
                        
                        # Look for the original token in the predictions
                        if original_token in top_tokens:
                            orig_idx = top_tokens.index(original_token)
                            heatmap_data[i, j] = top_probs[orig_idx]
                        else:
                            heatmap_data[i, j] = 0  # Token not in top predictions
                            
                    except (KeyError, IndexError):
                        heatmap_data[i, j] = 0
                else:
                    heatmap_data[i, j] = 0
        
        # Draw heatmap
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", 
                   xticklabels=[f"L{l}" for l in layers],
                   yticklabels=valid_tokens, ax=ax)
        
        ax.set_title("Probability of original token at each layer")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Token")
        
        return fig

    def visualize_top_predictions_heatmap(self, results, text_idx=0, k=1):
        """
        Create a heatmap visualization showing the top predicted token at each layer
        
        Args:
            results: Results from analyze_text
            text_idx: Which text to visualize if multiple were provided
            k: Number of top predictions to show in each cell
            
        Returns:
            matplotlib figure
        """
        tokens = results["tokens"][text_idx]
        layers = sorted([int(l.split('_')[1]) for l in results["layer_predictions"].keys()])
        
        # Skip special tokens
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, 
                        self.tokenizer.pad_token, self.tokenizer.mask_token,
                        '<s>', '</s>', '<pad>', '<mask>', '[CLS]', '[SEP]', '[PAD]', '[MASK]']:
                valid_tokens.append(token)
                valid_indices.append(i)
        
        if not valid_tokens:
            valid_tokens = tokens
            valid_indices = list(range(len(tokens)))
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(max(14, len(layers) * 1.2), max(10, len(valid_tokens) * 0.8)))
        
        # Prepare data for heatmap - textual representation with top tokens
        heatmap_data = np.zeros((len(valid_indices), len(layers)))
        heatmap_text = np.empty((len(valid_indices), len(layers)), dtype=object)
        
        for i, token_idx in enumerate(valid_indices):
            original_token = tokens[token_idx]
            
            for j, layer_idx in enumerate(layers):
                layer_name = f"layer_{layer_idx}"
                
                if layer_name in results["layer_predictions"]:
                    try:
                        # Get predictions for this token at this layer
                        layer_preds = results["layer_predictions"][layer_name]
                        top_tokens = layer_preds["top_tokens"][text_idx][token_idx]
                        top_probs = layer_preds["top_probs"][text_idx][token_idx]
                        
                        # Get the top k predictions with their probabilities
                        top_k_tokens = top_tokens[:k]
                        top_k_probs = top_probs[:k]
                        
                        # Format as text for display
                        text_entries = [f"{t}\n({p:.2f})" for t, p in zip(top_k_tokens, top_k_probs)]
                        heatmap_text[i, j] = "\n".join(text_entries)
                        
                        # Use the top token's probability for the heatmap color
                        heatmap_data[i, j] = top_probs[0]
                        
                    except (KeyError, IndexError):
                        heatmap_text[i, j] = "N/A"
                        heatmap_data[i, j] = 0
                else:
                    heatmap_text[i, j] = "N/A"
                    heatmap_data[i, j] = 0
        
        # Draw heatmap with text annotations
        sns.heatmap(heatmap_data, annot=heatmap_text, fmt="", cmap="Blues", 
                xticklabels=[f"L{l}" for l in layers],
                yticklabels=valid_tokens, ax=ax, cbar_kws={'label': 'Probability of top token'})
        
        # Adjust text size and rotation for better readability
        for text in ax.texts:
            text.set_fontsize(8)
        
        ax.set_title("Top token predictions across layers")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Original Token")
        
        plt.tight_layout()
        return fig


    def visualize_perplexity_per_layer(self, results):
        layer_predictions = results["layer_predictions"]
        pseudo_perplexities = []
        
        for layer, preds in layer_predictions.items():
            if isinstance(preds, torch.Tensor):
                logits = preds  # Use logits if already provided
            else:
                try:
                    logits = torch.tensor([pred["logits"] for pred in preds if isinstance(pred, dict) and "logits" in pred])
                except Exception as e:
                    print(f"Error processing predictions for layer {layer}: {e}")
                    continue

            # Apply softmax to get probabilities from logits
            probs = F.softmax(logits, dim=-1)
            
            # Calculate pseudo-perplexity using log-probabilities
            log_probs = torch.log(probs + 1e-9)  # Add epsilon to avoid log(0)
            token_pseudo_perplexities = torch.exp(-torch.sum(probs * log_probs, dim=-1))  # Using the token-wise pseudo-perplexity
            
            pseudo_perplexities.append(token_pseudo_perplexities.mean().item())

        # Create the figure using plt.figure() and then plot
        fig, ax = plt.subplots()
        ax.plot(range(len(pseudo_perplexities)), pseudo_perplexities, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Pseudo-Perplexity")
        ax.set_title("Pseudo-Perplexity at Each Layer")
        
        plt.tight_layout()
        return fig


    
    def visualize_representation_shift(self, results):
        hidden_states = results["hidden_states"]

        shifts = []
        layer_names = sorted(hidden_states.keys(), key=lambda x: int(x.split("_")[1]))  # Sort layers numerically
        for i in range(len(layer_names) - 1):
            layer_shift = torch.norm(hidden_states[layer_names[i + 1]] - hidden_states[layer_names[i]], dim=-1).mean().item()
            shifts.append(layer_shift)

        fig, ax = plt.subplots()
        ax.plot(range(len(shifts)), shifts, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Representation Shift")
        ax.set_title("Layer-wise Token Representation Shift")

        plt.tight_layout()
        return fig

    def compare_languages(self, text, languages, target_token_idx=None):
        """
        Compare logit lens results across different languages
        
        Args:
            text (str): Base text to translate
            languages (list): List of language codes
            target_token_idx (int, optional): Token position to analyze
            
        Returns:
            Visualization comparing languages
        """
        # This is a placeholder function - in a real application,
        # you would need to provide translations in each language
        # or use a translation service
        pass


def main():
    parser = argparse.ArgumentParser(description="Apply Logit Lens to multilingual models")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name or path (e.g., 'bert-base-multilingual-cased')")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name or path (defaults to model_name if not provided)")
    parser.add_argument("--text", type=str, default="This is a test sentence to analyze.",
                        help="Text to analyze")
    parser.add_argument("--lang", type=str, default=None,
                        help="Language code for the text (e.g., 'en', 'fr')")
    parser.add_argument("--visualization", type=str, default="grid",
                        choices=["line", "grid", "heatmap", "all"],
                        help="Visualization type")
    parser.add_argument("--output_dir", type=str, default="logit_lens_results",
                        help="Directory to save results")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top predictions to show")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize analyzer
    analyzer = LogitLensAnalyzer(args.model_name, args.tokenizer_name)
    
    # Analyze text
    results = analyzer.analyze_text(args.text, langs=args.lang)
    
    # Print some results
    print(f"\nAnalyzing: '{args.text}'")
    tokens = results["tokens"][0]
    print(f"Tokenized as: {tokens}")
    
    # Print predictions for a sample of layers
    layer_keys = sorted(results["layer_predictions"].keys())
    sample_layers = [layer_keys[0], layer_keys[len(layer_keys)//2], layer_keys[-1]]
    
    for layer_name in sample_layers:
        layer_idx = int(layer_name.split('_')[1])
        print(f"\n=== Layer {layer_idx} ===")
        
        # Show predictions for middle token
        mid_token_idx = len(tokens) // 2
        token = tokens[mid_token_idx]
        
        print(f"Middle token '{token}' predictions:")
        try:
            top_tokens = results["layer_predictions"][layer_name]["top_tokens"][0][mid_token_idx]
            top_probs = results["layer_predictions"][layer_name]["top_probs"][0][mid_token_idx]
            
            for t, p in zip(top_tokens, top_probs):
                print(f"  {t}: {p:.4f}")
        except KeyError:
            print("  No predictions available")
    
    figures = analyzer.visualize_all_predictions(results)
    
    # Save all figures
    for name, fig in figures.items():
        output_path = os.path.join(args.output_dir, f"{name}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {name} visualization to {output_path}")
    
    print(f"\nAll visualizations saved to {args.output_dir}")   

    # Save full results as text
    result_file = os.path.join(args.output_dir, "results_summary.txt")
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Text: {args.text}\n")
        f.write(f"Tokens: {tokens}\n\n")
        
        for layer_name in sorted(results["layer_predictions"].keys(), 
                               key=lambda x: int(x.split('_')[1])):
            layer_idx = int(layer_name.split('_')[1])
            f.write(f"=== Layer {layer_idx} ===\n")
            
            for token_idx, token in enumerate(tokens):
                f.write(f"Token '{token}' predictions:\n")
                try:
                    top_tokens = results["layer_predictions"][layer_name]["top_tokens"][0][token_idx]
                    top_probs = results["layer_predictions"][layer_name]["top_probs"][0][token_idx]
                    
                    for t, p in zip(top_tokens, top_probs):
                        f.write(f"  {t}: {p:.4f}\n")
                except (KeyError, IndexError):
                    f.write("  No predictions available\n")
                f.write("\n")
    
    print(f"Full results saved to {result_file}")

if __name__ == "__main__":
    main()