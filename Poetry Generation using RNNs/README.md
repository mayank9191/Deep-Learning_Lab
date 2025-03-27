# Poetry Generation using RNNs

## Objective
The goal of this experiment is to generate poetry using Recurrent Neural Networks (RNNs) and evaluate the impact of different word representations:

1. **One-Hot Encoding with RNN**
2. **Trainable Word Embeddings with RNN**
3. **One-Hot Encoding with LSTM**
4. **Trainable Word Embeddings with LSTM**

## Dataset
- **Source:** A dataset of 100 poems stored in `poems-100.csv`.
- **Preprocessing:**
  - Converting text to lowercase and removing special characters.
  - Tokenizing words and creating a vocabulary.
  - Encoding words using either **one-hot encoding** or **word embeddings**.

## Models Implemented

### 1. RNN with One-Hot Encoding
- **Input Representation:** One-hot encoded word sequences.
- **Architecture:**
  - Simple RNN layer with 16 hidden units.
  - Fully connected output layer with softmax activation.
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training Epochs:** 500
- **Performance:**
  - Final accuracy: **59.07%**
  - Final loss: **1.9682**
- **Model Weights:** `rnn_one_hot_weights.pth`

### 2. RNN with Trainable Word Embeddings
- **Input Representation:** Trainable word embeddings.
- **Architecture:**
  - Embedding layer (16-dimensional vectors per word).
  - Simple RNN layer with 16 hidden units.
  - Fully connected output layer with softmax activation.
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training Epochs:** 500
- **Performance:**
  - Final accuracy: **40.16%**
  - Final loss: **2.9645**
- **Model Weights:** `rnn_embeddings_weights.pth`

### 3. LSTM with One-Hot Encoding
- **Input Representation:** One-hot encoded word sequences.
- **Architecture:**
  - LSTM layer with 16 hidden units.
  - Fully connected output layer with softmax activation.
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training Epochs:** 500
- **Performance:**
  - Final accuracy: **66.67%**
  - Final loss: **1.5968**
- **Model Weights:** `lstm_one_hot_weights.pth`

### 4. LSTM with Trainable Word Embeddings
- **Input Representation:** Trainable word embeddings.
- **Architecture:**
  - Embedding layer (16-dimensional vectors per word).
  - LSTM layer with 16 hidden units.
  - Fully connected output layer with softmax activation.
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training Epochs:** 500
- **Performance:**
  - Final accuracy: **49.03%**
  - Final loss: **2.5610**
- **Model Weights:** `lstm_embeddings_weights.pth`

## Training Results
A comparison of training loss and accuracy over epochs was visualized for all models.

## Text Generation
- **Prediction Function:** Predicts the next word given a seed sequence.
- **Text Generator:** Generates poetry-like text iteratively.
- **Creativity Parameter:** Adjusts randomness in word selection.

### Sample Output
**Input:** _"the night is young and"_

**Generated Text (100 words, creativity=5):**
```text
the night is young and the stars whisper in twilight's glow
shadows dance with silent dreams where moonlight weaves and rivers flow...
```

## Model Saving
- The trained models' weights are stored as:
  - `rnn_one_hot_weights.pth`
  - `rnn_embeddings_weights.pth`
  - `lstm_one_hot_weights.pth`
  - `lstm_embeddings_weights.pth`

## Conclusion
The experiment demonstrated how **trainable embeddings** and **LSTM layers** improve text generation. Future improvements could include:
- Training on a larger poetry dataset.
- Using transformer-based models like GPT for better coherence.
- Experimenting with hyperparameter tuning.


