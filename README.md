# NanoGPT
# Table of Contents
- [Overview](#overview) 
- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [Code Explanation](#code-explanation)

# Overview
This repository contains a PyTorch implementation of a GPT-style language model trained on the works of Shakespeare. The model is designed to predict the next token in a sequence of text, allowing it to generate coherent text based on the patterns it learns from Shakespeare's writings. The implementation uses Transformer architecture components such as self-attention and feedforward layers to achieve effective language modeling and text generation capabilities.

# Requirements
- Python 3.x
- PyTorch

# Installation and Usage
- Create a virtual environment:
  ```
  python -m venv env
  ```
- Activate the virtual environment:
  ```
  .\env\Scripts\activate
  ```
- Install the Libraries:
  ```
  pip install -r requirements.txt
  ```
- Place your dataset of names in a text file named input.txt inside a data directory.
- Run:
  ```
  python nanogpt.py
  ```
- Deactivate the Environment:
  ```
  deactivate
  ```

# Code Explanation

1. Data Loading and Preparation
    - get_batch(split): Retrieves batches of data for training or validation.

2. Model Architecture
    - Head: Implements a single head of self-attention.
    - MultiHeadAttention: Manages multiple heads of self-attention in parallel.
    - FeedFoward: Defines a feedforward network for each transformer block.
    - Block: Represents a transformer block consisting of self-attention and feedforward layers.
    - GPTLanguageModel: Constructs the GPT-style language model using the Transformer architecture.

3. Training and Evaluation
    - estimate_loss(): Computes and evaluates loss on both training and validation sets.
    - Training Loop: Iteratively trains the model using batches of data, optimizing parameters based on computed loss.
  
4. Text Generation
    - generate(): Utilizes the trained model to generate new text based on a given initial context.
  
5. Execution and Output
    - Data Files: Input (input.txt) and Output (output.txt) files store text data and generated text respectively.
