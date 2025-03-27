# Deep Learning Lab Experiments

This repository contains the code for the experiments conducted in the Deep Learning Lab. The objective of these experiments is to explore different aspects of neural networks, including model architectures, activation functions, weight initialization techniques, and optimizers. Below is a brief overview of each experiment:

## Table of Contents
- [Experiment 1: MNIST Classification Model using only Numpy and Python](#experiment-1-mnist-classification-model-using-only-numpy-and-python)
- [Experiment 2: Neural Network for Linearly Separable Datasets](#experiment-2-neural-network-for-linearly-separable-datasets)
- [Experiment 3: Convolutional Neural Networks (CNNs) for Image Classification](#experiment-3-convolutional-neural-networks-cnns-for-image-classification)
- [Experiment 4: Recurrent Neural Networks (RNNs) for Text Generation](#experiment-4-poetry-generation-using-rnns)

## Experiment 1: MNIST Classification Model using only Numpy and Python

**Objective**:  
In this experiment, we built a simple neural network to classify handwritten digits from the MNIST dataset using only Numpy and Python. The goal was to implement the model from scratch to understand the fundamental concepts of forward propagation, backpropagation, and training without relying on high-level machine learning libraries.

**Key Concepts**:
- Building a neural network from scratch
- Implementing forward and backward propagation
- Training the network using gradient descent

## Experiment 2: Neural Network for Linearly Separable Datasets

**Objective**:  
This experiment focuses on training a neural network to classify linearly separable datasets (e.g., generated using sklearn's `make_moons` or `make_circles`). The experiment includes the following steps:
1. Train a simple neural network with no hidden layers and observe its failure to classify non-linearly separable data.
2. Add a hidden layer to the network and use activation functions like "ReLU" or "sigmoid" to observe the difference in performance.
3. Explore different non-linear datasets to observe the impact of hidden layers.

**Key Concepts**:
- Classification of linearly separable vs non-linearly separable data
- The impact of hidden layers and activation functions
- Experimenting with different dataset configurations

## Experiment 3: Convolutional Neural Networks (CNNs) for Image Classification

**Objective**:  
In this experiment, we implemented Convolutional Neural Networks (CNNs) to classify images from the **Cats vs. Dogs** and **CIFAR-10** datasets. We experimented with the following:
1. Different activation functions (e.g., ReLU, Sigmoid, Tanh)
2. Various weight initialization techniques (e.g., Xavier, He, and random initialization)
3. Different optimizers (e.g., Adam, SGD, RMSProp)

Additionally, we compared the performance of our best CNN model with a pretrained **ResNet-18** model on both the Cats vs. Dogs and CIFAR-10 datasets.

**Key Concepts**:
- CNN architecture for image classification
- Experimenting with different activation functions, weight initialization, and optimizers
- Fine-tuning models and comparing with pretrained networks like ResNet-18

# Experiment 4: Poetry Generation using RNNs

## Objective  
This experiment explores text generation using **Recurrent Neural Networks (RNNs)** trained on a dataset of poems. We evaluate different word representations:
1. One-Hot Encoding with RNN
2. Trainable Word Embeddings with RNN
3. One-Hot Encoding with LSTM
4. Trainable Word Embeddings with LSTM

## Key Concepts  
- Preprocessing text data for sequence modeling  
- Training RNN and LSTM-based text generation models  
- One-hot encoding vs. trainable word embeddings  
- Evaluating model performance and generating text samples  
