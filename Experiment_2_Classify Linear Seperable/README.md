# Neural Network for Linearly Separable and Non-Linearly Separable Datasets

This project demonstrates the application of neural networks to classify datasets that are both linearly separable and non-linearly separable. Initially, the network is trained on a linearly separable dataset, followed by testing on non-linearly separable datasets, such as "Moon" and "Circle". The project then enhances the neural network by adding a hidden layer to improve performance on non-linearly separable data.

## Overview

- **Linearly separable data**: The network is first trained on a simple dataset that is linearly separable using a neural network without hidden layers.
- **Non-linearly separable data**: The network is tested on more complex datasets like the "Moon" and "Circle", where a simple neural network fails to classify correctly.
- **Improved model**: To handle non-linearly separable data, a hidden layer is added to the neural network, improving its classification performance.

## Datasets

1. **Linearly Separable Dataset**: A synthetic dataset generated using `make_classification` from `sklearn.datasets`.
2. **Non-Linearly Separable Datasets**:
   - **Moon Dataset**: Generated using `make_moons` from `sklearn.datasets`.
   - **Circle Dataset**: Generated using `make_circles` from `sklearn.datasets`.

## Model Architecture

- **Initial Model**: A simple neural network with input and output layers but no hidden layers, trained on the linearly separable dataset.
- **Enhanced Model**: A more complex neural network with an added hidden layer, used to handle non-linearly separable datasets.

## Results

- The initial network performs well on the linearly separable dataset but struggles on the non-linearly separable datasets.
- After adding a hidden layer, the model performs significantly better on non-linearly separable data, demonstrating the power of more complex neural network architectures.

