# MiniGPT using Small Language Model (SLM)

## 📌 Overview

This project implements a **MiniGPT-style language model from scratch using PyTorch**.
It is a **decoder-only Transformer** trained for **character-level text generation**.

The goal of this project is to understand how modern language models work internally, instead of relying on high-level libraries.

---

## Key Concepts Covered

### 1. Transformer Architecture

* Uses a **decoder-only Transformer**
* Stacks multiple Transformer blocks
* Each block contains:

  * Self-attention
  * Feedforward network
  * Residual connections
  * Layer normalization

---

### 2. Self-Attention (Implemented from Scratch)

The core idea of attention is to let each token focus on relevant past tokens.

Steps implemented:

* Compute **Query (Q), Key (K), Value (V)** using linear layers
* Calculate attention scores using:
  [
  \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
  ]
* Multiply with values to get output

---

### 3. Causal Masking

* Prevents the model from seeing future tokens
* Implemented using a **lower triangular matrix**
* Makes the model **autoregressive (GPT-style)**

---

### 4. Multi-Head Attention

* Splits attention into multiple heads
* Each head learns different relationships
* Outputs are concatenated and projected

---

### 5. Character-Level Language Model

* Text is processed at **character level**
* Each character is mapped to an integer
* Model predicts the **next character in sequence**

---

## Features

* Manual implementation of attention (no built-in transformer modules)
* Modular code structure
* Works on CPU and GPU
* Simple and easy to understand



## 📊 What the Model Does

* Learns patterns from text data
* Generates new text one character at a time
* Demonstrates how GPT-like models work internally

---

## 🚀 Future Improvements

* Use word-level or BPE tokenization
* Add better sampling methods (top-k, temperature)
* Train on larger datasets
* Optimize training and performance

---

## 📌 Note

This is a **learning-focused implementation** aimed at understanding the fundamentals of Transformers and attention mechanisms.

---
