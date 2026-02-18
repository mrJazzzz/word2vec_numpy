# Word2Vec from Scratch (Skip-Gram with Negative Sampling)

This project is a **from-scratch implementation of Word2Vec** in Python using **NumPy**. It uses the **skip-gram architecture** with **negative sampling**, similar to the original Word2Vec model by Mikolov et al. This project is designed for educational purposes and gives insight into how word embeddings are learned from raw text data.

---

## Features

- Preprocessing of text, including:
  - Removing rare words (`min_count`)
  - Subsampling frequent words
  - Preparing negative sampling probabilities
- Skip-gram model with negative sampling
- Adjustable hyperparameters:
  - Embedding dimension (`d`)
  - Window size (`m`)
  - Negative samples per target (`K`)
  - Learning rate and epochs
- Monitoring of training progress
- Saves trained embeddings to `.npz` file

---

## Installation

This project only requires Python 3 and **NumPy**. You can install NumPy via pip if you don’t already have it:

```bash
pip install numpy
```
---

## Link to dataset

The dataset is Text8, that can be found [here](https://www.kaggle.com/datasets/yorkyong/text8-zip).
