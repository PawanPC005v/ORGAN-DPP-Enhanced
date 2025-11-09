#  ORGAN-DPP: Feature-Based Determinantal Point Process for Memory-Efficient Molecular Generation

### B.Tech Final Year Project — Department of Artificial Intelligence & Machine Learning  
**Manakula Vinayagar Institute of Technology, Pondicherry University**

---

##  Project Overview
**ORGAN-DPP (Objective-Reinforced Generative Adversarial Network with Determinantal Point Process)** is a deep generative framework designed for **AI-driven drug discovery**.  
It integrates:
- **Feature-based DPPs** for memory-efficient molecular diversity.
- **Curriculum Learning** for structured and stable training.
- **Adaptive Temperature Annealing** for balanced exploration–exploitation.

The model efficiently generates **novel, valid, and drug-like molecules** on **consumer-grade GPUs**.

---

##  Key Features
-  **85% memory reduction** (O(nd) vs. O(n²)) using feature-space DPP.
-  **Three-stage curriculum**: Validity → Drug-likeness → Synthesis optimization.
-  **Adaptive temperature annealing** for exploration balance.
-  **State-of-the-art results**: 94% validity, 0.88 diversity, 0.61 QED.
-  **Fully open-source**, modular, and runs on Google Colab / Hugging Face Spaces.

---

## System Architecture

| Module | Description |
|--------|--------------|
| **Generator** | Temperature-controlled LSTM for SMILES sequence generation |
| **Discriminator** | CNN-based validity evaluator |
| **Feature-based DPP** | Ensures structural diversity efficiently |
| **Curriculum Scheduler** | Manages 3-stage adaptive training |
| **Visualizer** | RDKit + Matplotlib for molecule rendering and metrics |

## Dataset Description

| File | Description |
|------|--------------|
| **train.txt** | Training SMILES (ZINC subset, tokenized) |
| **test.txt** | Testing SMILES |
| **vocab.txt** | List of token vocabulary (e.g., atoms, bonds, brackets) |
| **zinc_clean.smi** | Cleaned dataset generated via RDKit preprocessing |

---

## Tech Stack
- **Language:** Python 3.8+
- **Frameworks:** PyTorch, RDKit, NumPy, SciPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Interface:** Gradio / Flask
- **Deployment:** Google Colab, Hugging Face Spaces, Docker

----

## Performance Results

| Metric | ORGAN | ORGAN-DPP | Improvement |
|:--------|:------:|:----------:|:-------------:|
| **Validity** | 0.89 | 0.94 | **+5.6%** |
| **Diversity** | 0.75 | 0.88 | **+17.3%** |
| **Uniqueness** | 0.82 | 0.91 | **+11%** |
| **QED** | 0.52 | 0.61 | **+17%** |
| **Training Time** | 10.5 h | 8.2 h | **−22%** |
| **Memory Usage** | 14.8 GB | 5.1 GB | **−85%** |
