# On Multilingual Encoder Language Model Compression for Low-Resource Languages

[![arXiv](https://img.shields.io/badge/arXiv-2505.16956v1-b31b1b.svg)](https://arxiv.org/abs/2505.16956)

In this paper, we combine two-step knowledge distillation, structured pruning, truncation, and vocabulary trimming for extremely compressing multilingual encoder-only language models for low-resource languages. Our novel approach systematically combines existing techniques and takes them to the extreme, reducing layer depth, feed-forward hidden size, and intermediate layer embedding size to create significantly smaller monolingual models while retaining essential language-specific knowledge. We achieve compression rates of up to 92% with only a marginal performance drop of 2-10% in four downstream tasks, including sentiment analysis, topic classification, named entity recognition, and part-of-speech tagging, across three low-resource languages. Notably, the performance degradation correlates with the amount of language-specific data in the teacher model, with larger datasets resulting in smaller performance losses. Additionally, we conduct extensive ablation studies to identify best practices for multilingual model compression using these techniques.

## Recent Work on Distillation (*through a multilingual lense*)

### Distillation's Main Idea
The student model is trained with the loss for the objective at hand (masked language modeling (MLM) for generic distillation and specific tasks for task-based distillation), while also forcing the predictions to be identical to the teacher model.

### Key Papers

#### **DistilBERT** ([HuggingFace, 2019](https://arxiv.org/pdf/1910.01108#page=3.02))
- **Student architecture:** Better to *reduce the number of layers* than sequence length
- **Student initialization:** *Initialize the student from the teacher* by taking one layer out of two
- **Distillation process:** *Distill on very large batches* leveraging gradient accumulation (up to 4 examples per batch) using dynamic masking
- **Data and compute:** English Wikipedia and Toronto Book Corpus; trained on 8 16GB V100 GPUs for ~90 hours

#### **Multilingual Distilled BERT** ([HuggingFace, 2020](https://arxiv.org/pdf/1910.01108#page=3.02))
- 6 layers, 768 dimension, 12 heads, 134M parameters (vs. 177M for mBERT-base)
- Trained on concatenation of Wikipedia in 104 languages

#### **Advanced Distillation** - Patient Knowledge Distillation ([Sun et al., 2019](https://arxiv.org/abs/1908.09355))
- Learn from intermediate layers, not only the final output layer

#### **Distilling Monolingual Models from mBERT** ([Singh et al., 2022](https://aclanthology.org/2022.coling-1.391.pdf))
- **First work distilling monolingual models**
- Distillation loss = NLL, Cosine loss = Directional similarity, MLM loss = Standard cross-entropy
- **Reduce vocabulary of the student model** post-distillation
- **Initialization from the teacher model improves performance**
- **Fine-tuned on downstream tasks (sentiment, topic classification, POS, NER)**

#### **Distilling Efficient Language-Specific Models for Cross-Lingual Transfer** ([Ansell et al., 2023](https://aclanthology.org/2023.findings-acl.517.pdf))
- **Bilingual distillation:** Only source and target language
- Two-phase training: 
  1. General phase - align hidden representations
  2. Task-specific phase - fine-tune student with task-adapted teacher
- **Lottery-Ticket Sparse FineTuning (LT-SFT)** for efficient multi-task training

#### **The Privileged Students: On the Value of Initialization in Multilingual Knowledge Distillation** ([Wibowo et al., 2024](https://arxiv.org/abs/2406.16524))
- **Initialization from fine-tuned teacher contributes the most**
- **MSE instead of KL Divergence** → Faster convergence and higher performance


```bibtex
@misc{gurgurov2025multilingualencoderlanguagemodel,
      title={On Multilingual Encoder Language Model Compression for Low-Resource Languages}, 
      author={Daniil Gurgurov and Michal Gregor and Josef van Genabith and Simon Ostermann},
      year={2025},
      eprint={2505.16956},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16956}, 
}
```
