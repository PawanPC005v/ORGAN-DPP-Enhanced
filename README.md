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
![Architecture Diagram](docs/architecture.png)

| Module | Description |
|--------|--------------|
| **Generator** | Temperature-controlled LSTM for SMILES sequence generation |
| **Discriminator** | CNN-based validity evaluator |
| **Feature-based DPP** | Ensures structural diversity efficiently |
| **Curriculum Scheduler** | Manages 3-stage adaptive training |
| **Visualizer** | RDKit + Matplotlib for molecule rendering and metrics |

---

## Tech Stack
- **Language:** Python 3.8+
- **Frameworks:** PyTorch, RDKit, NumPy, SciPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Interface:** Gradio / Flask
- **Deployment:** Google Colab, Hugging Face Spaces, Docker

---

## 📂 Repository Structure

