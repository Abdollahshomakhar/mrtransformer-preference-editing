# MrTransformer with Preference Editing

A PyTorch implementation of **MrTransformer**, a sequential recommendation model enhanced with **Preference Editing** and **Dynamic Preference Weighting**.

This project investigates how disentangled user preferences can be learned from sequential user-item interactions and dynamically combined according to the user's current behavioral context.

## Overview

Sequential recommendation aims to predict a user's next interaction based on their previous interaction history.

In this project, the original MrTransformer architecture is implemented and extended with:

* **Multiple Preference Representations** for modeling different aspects of user interests
* **Preference Editing (PE)** for learning shared and distinct preference information
* **Dynamic Preference Weighting (DPW)** for adapting preference importance according to the current context
* **Coverage Loss** to encourage diversity among learned preference representations
* **Self-Supervised Preference Learning** using preference recombination and common-preference consistency

The complete training pipeline includes pretraining, preference-aware fine-tuning, and ranking-based evaluation.

## Model Architecture

The model consists of the following main components:

```text
User Interaction Sequence
          │
          ▼
   Item Embeddings
          │
          ▼
   Transformer Encoder
          │
          ├───────────────┐
          ▼               ▼
 Preference Tokens    Context Representation
          │               │
          └───────┬───────┘
                  ▼
     Dynamic Preference Weighting
                  │
                  ▼
        User Representation
                  │
                  ▼
       Item Prediction Scores
```

During preference-aware training, two sequences are processed and their preference representations are compared through the Preference Editing module.

## Main Components

### 1. MrTransformer

The base sequential recommendation architecture uses Transformer-based sequence modeling together with multiple learnable preference representations.

### 2. Preference Editing

Preference Editing compares preference representations between two sequences and separates:

* Common preference information
* Sequence-specific preference information

The resulting representations are used as an additional self-supervised learning signal.

### 3. Dynamic Preference Weighting

Dynamic Preference Weighting determines the contribution of each learned preference according to the current user context.

Instead of assigning the same importance to all preference representations, the model dynamically calculates preference weights for each sequence.

### 4. Coverage Loss

Coverage regularization encourages different preference representations to attend to different parts of the sequence, reducing redundancy between preferences.

### 5. Self-Supervised Learning

The fine-tuning stage incorporates preference-level self-supervision through:

* Preference recombination consistency
* Common preference similarity

The final objective combines recommendation, coverage, and self-supervised losses.

## Dataset

The experiments use the **MovieLens-100K** dataset.

The interaction sequences are ordered chronologically for each user.

The evaluation protocol follows a sequential next-item recommendation setting:

* The last interaction is used as the test target.
* The second-to-last interaction is used for validation.
* Earlier interactions are used for training.
* Negative items are sampled during evaluation.
* Ranking metrics are calculated using 100 negative candidates.

## Training Pipeline

The training process consists of two stages.

### Stage 1 — Pretraining

The base MrTransformer model is pretrained using the recommendation objective together with the coverage-related objective.

Several learning rates are evaluated to determine an appropriate initialization for the subsequent fine-tuning stage.

### Stage 2 — Preference-Aware Fine-Tuning

The pretrained model is fine-tuned using:

```text
Recommendation Loss
        +
Coverage Loss
        +
Preference Self-Supervised Loss
```

The fine-tuning stage incorporates Preference Editing and Dynamic Preference Weighting.

## Evaluation Metrics

The model is evaluated using standard top-K ranking metrics:

### Recall

* Recall@5
* Recall@10
* Recall@20

### Mean Reciprocal Rank

* MRR@5
* MRR@10
* MRR@20

### Normalized Discounted Cumulative Gain

* NDCG@5
* NDCG@10
* NDCG@20

These metrics evaluate both whether the target item appears in the top-K recommendations and how highly it is ranked.

## Hyperparameters

The main experimental configuration includes:

| Parameter                 | Value |
| ------------------------- | ----: |
| Maximum Sequence Length   |    50 |
| Hidden Dimension          |    64 |
| Transformer Layers        |     2 |
| Attention Heads           |     2 |
| Number of Preferences     |     3 |
| Dropout                   |   0.2 |
| Batch Size                |   128 |
| Fine-tuning Learning Rate |  1e-3 |
| Weight Decay              |  1e-4 |
| Coverage Weight           |   0.1 |
| SSL Recombination Weight  |   0.1 |
| SSL Common Weight         |   0.1 |
| Negative Samples          |   100 |

## Learning Rate Experiment

Different pretraining learning rates are evaluated:

```text
1e-4
5e-4
1e-3
2e-3
5e-3
```

The pretrained models are subsequently fine-tuned using the same fine-tuning configuration so that the effect of the pretraining learning rate can be investigated independently.

## Project Structure

```text
mrtransformer-preference-editing/
│
├── eval test data for plot.ipynb
├── images and training.ipynb
├── toys training.ipynb
├── x.ipynb
└── README.md
```

> The exact file structure may vary depending on the current experimental version of the project.

## Reproducibility

To reproduce the experiments:

```bash
git clone https://github.com/Abdollahshomakhar/mrtransformer-preference-editing.git

cd mrtransformer-preference-editing

pip install -r requirements.txt
```

Prepare the MovieLens-100K dataset, configure the dataset path, and run the pretraining and fine-tuning stages.

Example:

```bash
python pretrain.py
python finetune.py
python evaluate.py
```

## Research Purpose

This repository was developed as part of a master's thesis research project on **sequential recommendation and preference-aware representation learning**.

The primary objective is to study whether modeling multiple user preferences and dynamically weighting them according to the current context can improve sequential recommendation performance.

## Citation

If you use this implementation, experimental results, or code in your research, please cite this repository.

```bibtex
@software{ghorbani_mrtransformer_preference_editing,
  author  = {Ehsan Ghorbani},
  title   = {MrTransformer with Preference Editing},
  year    = {2026},
  url     = {https://github.com/Abdollahshomakhar/mrtransformer-preference-editing}
}
```

## License

This project is intended for research and educational purposes.
  
