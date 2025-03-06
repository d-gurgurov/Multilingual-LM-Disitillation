# Knowledge Distillation for Multilingual Transformers

## Recent Literature on Distillation

### Distillation's Main Idea
The student model is trained with the loss for the objective at hand (masked language modeling (MLM) for generic distillation and specific tasks for task-based distillation), while also forcing the predictions to be identical to the teacher model.

### Key Papers

#### **DistilBERT** ([HuggingFace, 2019](https://arxiv.org/pdf/1910.01108#page=3.02))
- **Student architecture:** Better to *reduce the number of layers* than sequence length
- **Student initialization:** *Initialize the student from the teacher* by taking one layer out of two
- **Distillation process:** *Distill on very large batches* leveraging gradient accumulation (up to 4 examples per batch) using dynamic masking
- **Data and compute:** English Wikipedia and Toronto Book Corpus; trained on 8 16GB V100 GPUs for ~90 hours

#### **Multilingual Distilled BERT** ([HuggingFace, 2020])
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

## Experiments

### **Baselines**
1. **Train students without supervision** (layer-reduced mBERT)
2. **Fine-tune full models** (mBERT as is)

### **Experiment 1: KL vs. MSE for Distillation Loss**
- **KL (with/without temperature logit softening)**
- **MSE (with/without temperature logit softening)**
- Results: **MSE without logit softening performs best**
- Supporting evidence: [Kim et al. (2021)](https://arxiv.org/pdf/2201.00558), [Nityasya et al. (2022)](https://arxiv.org/pdf/2201.00558)

### **Experiment 2: Teacher vs. Random Initialization**
- **Teacher initialization performs significantly better** than random
- Supporting evidence: [Sun et al., 2019](https://arxiv.org/abs/1908.09355), [Singh et al., 2022](https://aclanthology.org/2022.coling-1.391.pdf), [Wibowo et al. (2024)](https://arxiv.org/abs/2406.16524)

### **Experiment 3: Varying Loss Weights (α-values for Distillation Loss Influence)**
- Higher α → Lower distillation influence
- Lower α → Higher distillation influence
- **Best performance achieved with α = 0.2** (strong distillation weight)

### **Experiment 4: Model Width Reduction**
- Reducing **layers by 2, 4, and 6**
- **Distilled models initially perform better, but saturate**
- Similar findings in [Cho et al., 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf?utm_source=chatgpt.com)

### **Experiment 5: Vocabulary Reduction**
- Inspired by [Abdaoui et al. (2020)](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2010.05609&hl=en&sa=T&oi=gsr-r&ct=res&cd=0&d=2362710075020124582&ei=8Tq_Z9aBBuehieoPzq2wqAs&scisig=AFWwaeYZthWpKvrRHbsUXbwtOKdJ)
- Reducing **embedding layer to target language tokens only**

### **Experiment 6: Different Objective Functions**
- **KL + MLM** ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531))
- **MSE + MLM** ([Kim et al., 2021](https://arxiv.org/pdf/2201.00558))
- **KL + Cosine Similarity + MLM** ([Singh et al., 2022](https://aclanthology.org/2022.coling-1.391.pdf))
- **Multilayer loss (TinyBERT objective)** ([Jiao et al., 2020](https://aclanthology.org/2020.findings-emnlp.372/))

### **Experiment 7: Fine-Tuning on Downstream Tasks**
- **Topic classification** with best student model
- Comparing **distilled models vs. full fine-tuned models**
