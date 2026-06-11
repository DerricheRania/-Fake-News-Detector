# 🔍 Fake News Detector

> An AI-powered news classification system built on a fine-tuned **DistilBERT** transformer that detects whether a news article is real or fake, with a confidence score and word-level explainability.

## Overview

This project builds an end-to-end fake news detection system using Natural Language Processing and Transfer Learning. Given the title and body of a news article, the model outputs:

- A **classification label** : `FAKE` or `REAL`
- A **confidence score** : probability for each class
- An **explanation** : which words in the article most influenced the decision (powered by LIME)

The model was fine-tuned on the **ISOT Fake News Dataset**, achieving **99.90% accuracy** and a **perfect ROC-AUC of 1.0000** on the held-out test set.

---

## Dataset

**ISOT Fake News Dataset** — University of Victoria  
Source: [Kaggle : emineyetm/fake-news-detection-datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

| Split | Samples |
|-------|---------|
| Total | 44,898 |
| Fake  | 23,481  |
| Real  | 21,417  |
| Train | 31,428 (70%) |
| Validation | 6,735 (15%) |
| Test  | 6,735 (15%) |

The dataset contains two CSV files — `Fake.csv` and `True.csv` each with columns: `title`, `text`, `subject`, `date`. Real articles were scraped from Reuters; fake articles from PolitiFact and other flagged sources.

The split is **stratified**, ensuring balanced class ratios across all three sets. Shuffling is seeded at `42` for full reproducibility.

---

## Project Structure

```
fake-news-detector/
│
├── fake_news_detector.ipynb       # Main notebook (full pipeline)
├── best_fake_news_model.pt        # Best model checkpoint (by Val F1)
├── fake_news_detector_model/      # Saved HuggingFace model + tokenizer
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
│
├── eda_plots.png                  # EDA visualizations
├── training_curves.png            # Loss & accuracy over epochs
├── evaluation_plots.png           # Confusion matrix + ROC curve
└── lime_explanation.png           # LIME word importance chart
```

---

## Pipeline

```
Raw CSVs (Fake.csv + True.csv)
        │
        ▼
  Data Loading & Labeling
  (label 0 = Fake, 1 = Real)
        │
        ▼
  Exploratory Data Analysis
  (class balance, text length, subject distribution)
        │
        ▼
  Text Preprocessing
  (remove datelines, URLs, whitespace)
  → Combine: "[TITLE] [SEP] [first 400 words of body]"
        │
        ▼
  Train / Val / Test Split
  (70% / 15% / 15%, stratified)
        │
        ▼
  DistilBertTokenizerFast
  (max_length=256, padding, truncation)
        │
        ▼
  Fine-tune DistilBertForSequenceClassification
  (3 epochs, AdamW, linear warmup scheduler)
        │
        ▼
  Evaluation
  (Accuracy, F1, ROC-AUC, Confusion Matrix)
        │
        ▼
  LIME Explainability
  (word-level feature importance)
        │
        ▼
  Interactive Inference: predict_article(title, body)
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Base model | `distilbert-base-uncased` |
| Task head | Linear classifier (2 classes) |
| Total parameters | 67.0M |
| Trainable parameters | 67.0M (fully fine-tuned) |
| Max token length | 256 |
| Optimizer | AdamW (`lr=2e-5`, `weight_decay=0.01`) |
| Scheduler | Linear warmup (10% of steps) → linear decay |
| Gradient clipping | `max_norm=1.0` |
| Batch size | 16 |
| Epochs | 3 |
| Device | CUDA (GPU) |
| Random seed | 42 |

The input combines the article **title** and the **first 400 words** of the body, separated by a `[SEP]` token, before being passed to the tokenizer. This gives the model both the headline signal and enough body context without exceeding token limits.

---

## Results

### Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|-----------|-----------|----------|---------|--------|
| 1/3   | 0.0712    | 97.37%    | 0.0068   | 99.82%  | 0.9982 |
| 2/3   | 0.0048    | 99.88%    | 0.0061   | 99.88%  | 0.9988 |
| 3/3   | 0.0010    | 99.99%    | 0.0064   | 99.90%  | **0.9990** ✅ |

The best checkpoint (epoch 3, Val F1 = 0.9990) was saved and used for final evaluation.

### Test Set Evaluation

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.90%** |
| **F1 Score (weighted)** | **0.9990** |
| **ROC-AUC** | **1.0000** |
| Test Loss | 0.0042 |

### Per-Class Report

```
              precision    recall  f1-score   support

        FAKE       1.00      1.00      1.00      3523
        REAL       1.00      1.00      1.00      3212

    accuracy                           1.00      6735
   macro avg       1.00      1.00      1.00      6735
weighted avg       1.00      1.00      1.00      6735
```

The model achieves near-perfect precision and recall on both classes, with only ~7 misclassified articles out of 6,735 in the test set.

---

## Explainability

The project implements **LIME** (Local Interpretable Model-Agnostic Explanations) to answer the question: *"Why did the model flag this article as fake?"*

LIME perturbs the input text, observes how predictions change, and identifies which words most influenced the decision. This is critical for real-world deployment where users need to trust and understand the model's reasoning.

### Example — Fake Article

**Input title:** `"NASA Confirms Earth Will Experience 6 Days of Total Darkness in December Due to Solar Storm"`

**Prediction:** 🔴 FAKE : 100.00% confidence

**Top influential words:**

| Rank | Word | Direction |
|------|------|-----------|
| 1 | `this` | ▲ Pushes FAKE |
| 2 | `at` | ▲ Pushes FAKE |
| 3 | `rare` | ▲ Pushes FAKE |
| 10 | `memo` | ▲ Pushes FAKE |

Words like `"rare"`, `"memo"`, and `"this"` (used in call-to-action phrasing like *"Share this before it gets deleted"*) contributed to the FAKE classification.

### Example — Real Article

**Input title:** `"Federal Reserve Holds Interest Rates Steady Amid Cooling Inflation Data"`

**Prediction:** 🟢 REAL — 100.00% confidence

---

## Inference Demo

The notebook provides a ready-to-use `predict_article()` function:

```python
result = predict_article(
    title="NASA Confirms Earth Will Experience 6 Days of Total Darkness",
    body="NASA has officially confirmed that Earth will experience...",
    explain=True   # Set False to skip LIME for speed
)

# Output:
# 🔴 Prediction  : FAKE
# 📊 Confidence   : 100.00%
#    ├─ P(FAKE)   : 100.00%
#    └─ P(REAL)   :   0.00%
```

To reload the saved model in a new environment:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

model     = DistilBertForSequenceClassification.from_pretrained('./fake_news_detector_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./fake_news_detector_model')
model.eval()
```

---

## How to Run

### 1. Clone / download the notebook

Place `fake_news_detector.ipynb` in your working directory.

### 2. Download the ISOT dataset

From [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets), download and extract `Fake.csv` and `True.csv` into the same directory. Alternatively, configure the Kaggle CLI and the notebook will auto-download.

### 3. Install dependencies

```bash
pip install transformers datasets torch scikit-learn pandas numpy matplotlib seaborn lime tqdm
```

### 4. Run the notebook

Open in Jupyter or Kaggle and run all cells top to bottom.

> **GPU strongly recommended.** Training takes ~15–20 minutes on GPU, ~1–2 hours on CPU.

---

## Requirements

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Deep learning framework |
| `transformers >= 4.30` | DistilBERT model & tokenizer |
| `scikit-learn` | Metrics, train/test split |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualizations |
| `lime` | Explainability |
| `tqdm` | Progress bars |

---

## Author

Built as an NLP classification project using the ISOT Fake News Dataset and HuggingFace Transformers

---

*Model achieves 99.90% accuracy and ROC-AUC of 1.0000 on the ISOT test set using DistilBERT fine-tuned for 3 epochs.*
