# Masked Word Prediction NLP Model

## Overview
This project implements a deep learning model for masked word prediction, a natural language processing task where the model predicts missing words in sentences. The current version achieves a 28% accuracy rate, with ongoing efforts to improve performance.

## Model Architecture
The model uses a recurrent neural network architecture with the following components:
- Pre-trained GloVe word embeddings (100-dimensional)
- Bidirectional GRU layers for capturing context from both directions
- Layer normalization and dropout for regularization
- Multiple stacked GRU layers for deeper feature extraction
- Dense output layer with softmax activation

## Technical Details
- **Framework**: TensorFlow/Keras
- **Word Embeddings**: GloVe 6B 100d
- **Sequence Length**: 256 tokens (padded)
- **Vocabulary Size**: 30,000 tokens
- **Training Strategy**: Masked language modeling approach
- **Current Accuracy**: 28%

## Data Processing
The model processes input sentences by:
1. Tokenizing text using Keras Tokenizer
2. Padding sequences to uniform length
3. Creating training examples by masking individual words
4. Using the surrounding context to predict the masked word

## Training
The model is trained with:
- Adam optimizer (learning rate: 0.001)
- Sparse categorical cross-entropy loss
- Early stopping based on validation loss
- Batch size of 64
- Maximum 10 epochs

## Files
- `Train Data.csv`: Contains training sentences
- `Test Datas.csv`: Contains test sentences with masked words
- `glove.6B.100d.txt`: Pre-trained GloVe word embeddings
- `gru_model.h5`: Saved model weights
- `submission.csv`: Generated predictions for evaluation

## Future Improvements
Considering the current 28% accuracy, potential improvements include:
- Experimenting with larger embedding dimensions
- Adding attention mechanisms
- Increasing model capacity (more layers/units)
- Hyperparameter optimization
- Data augmentation techniques
- Alternative architectures (Transformer-based models)

## Requirements
- TensorFlow 2.x
- NumPy
- Pandas
- GloVe word embeddings

## Usage
1. Download GloVe embeddings
2. Prepare training and test data in required CSV format
3. Run the training script
4. Generate predictions on masked sentences
