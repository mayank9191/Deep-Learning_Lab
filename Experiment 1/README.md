## Experiment 1: MNIST Classification Model using only Numpy and Python

### Objective:
The goal of this experiment is to build a simple neural network model to classify handwritten digits from the **MNIST** dataset using only **Numpy** and **Python**. This experiment aims to understand the fundamental concepts of neural networks, including **forward propagation**, **backpropagation**, and **training** without relying on any high-level deep learning frameworks like TensorFlow or Keras.

By the end of this experiment, you will have gained an in-depth understanding of how neural networks work from scratch and how various mathematical operations, such as matrix multiplications, gradients, and weight updates, contribute to training a model.

### Key Concepts:
- **Forward Propagation**: Passing input data through the network layers to compute the predicted output.
- **Backpropagation**: Calculating the gradients and updating weights to minimize the loss function.
- **Gradient Descent**: Optimizing the model by updating weights in the direction of the negative gradient.
- **Activation Functions**: Using non-linear functions (like Sigmoid or ReLU) to introduce non-linearity into the model.

### Dataset:
The **MNIST dataset** is a large collection of handwritten digits (0-9). It consists of:
- 60,000 training images
- 10,000 testing images
Each image is a 28x28 pixel grayscale image, flattened into a vector of length 784.

### Model Architecture:
- **Input Layer**: A 784-dimensional vector (representing the flattened 28x28 image).
- **Hidden Layer**: A fully connected layer with 128 neurons and ReLU activation.
- **Output Layer**: A softmax layer with 10 neurons (one for each digit from 0 to 9).

### Training:
- **Loss Function**: Cross-entropy loss, used for multi-class classification.
- **Optimizer**: Gradient descent to minimize the loss function.

### Results and Observations:

After training the MNIST classification model for 250 epochs, the following results were observed:

- **Training Accuracy**: The training accuracy steadily improved as the model learned from the data, starting from around **12.48%** at epoch 1 and reaching **91.79%** by epoch 250.
  
- **Test Accuracy**: The model's test accuracy also improved consistently, reaching **92.21%** at the end of training, indicating that the model generalized well on unseen data.

#### Epoch-wise Accuracy:
- **Epoch 1**: 
  - Train Accuracy = **12.48%**
  - Test Accuracy = **35.24%**

- **Epoch 50**: 
  - Train Accuracy = **85.34%**
  - Test Accuracy = **86.15%**

- **Epoch 100**: 
  - Train Accuracy = **89.30%**
  - Test Accuracy = **89.81%**

- **Epoch 150**: 
  - Train Accuracy = **90.43%**
  - Test Accuracy = **90.95%**

- **Epoch 200**: 
  - Train Accuracy = **91.22%**
  - Test Accuracy = **91.65%**

- **Epoch 250**: 
  - Train Accuracy = **91.79%**
  - Test Accuracy = **92.21%**

### Conclusion:
The model performed well and achieved high accuracy on the MNIST dataset, demonstrating the effectiveness of the simple neural network we implemented. The increase in accuracy over time reflects successful learning through gradient descent, with both training and test accuracies stabilizing toward the end of the training process. The model was able to generalize well, as indicated by the minimal gap between training and test accuracy.
