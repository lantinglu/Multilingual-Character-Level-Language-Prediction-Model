
# Multilingual Character-Level Language Prediction Model

A multilingual character-level language modeling project for next-character prediction.  
This project builds a hybrid prediction framework that combines a Transformer-based character language model with an N-gram statistical language model to improve robustness in multilingual, low-resource, and noisy text scenarios.

## Project Overview

Next-character prediction is useful in text input systems, auto-completion, spelling correction, assistive communication, and multilingual text generation. Unlike word-level language models, character-level models can naturally support multiple languages without requiring language-specific tokenizers or vocabularies.

This project focuses on predicting the next character given an input text prefix. The model is designed to handle multilingual data and improve prediction stability by combining neural contextual modeling with statistical local pattern modeling.

## Key Features

- Built a multilingual character-level next-character prediction model
- Combined a Transformer language model with an N-gram language model
- Used weighted score fusion to balance contextual understanding and statistical robustness
- Supported multilingual text inputs at the character level
- Implemented an end-to-end training and inference pipeline
- Applied AdamW optimization and cosine annealing with warmup for stable training
- Evaluated model performance through prediction accuracy and experimental comparison

## Model Architecture

The project uses a hybrid modeling framework:

1. **Character-Level Tokenization**  
   Input text is processed at the character level. Each character is mapped to an index in the vocabulary.

2. **Transformer Language Model**  
   A character-level Transformer captures long-range contextual dependencies from the input prefix.

3. **N-gram Language Model**  
   A statistical N-gram model captures local character patterns and provides stable predictions, especially in low-resource or noisy contexts.

4. **Weighted Fusion**  
   The final prediction score combines Transformer probabilities and N-gram probabilities:

   ```text
   P_final = α * P_transformer + (1 - α) * P_ngram
