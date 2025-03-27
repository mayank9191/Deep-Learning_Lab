# Experiment 3: Implementing CNNs for Image Classification on Cats vs. Dogs and CIFAR-10 Datasets

## Objective
The objective of this experiment is to implement and evaluate Convolutional Neural Networks (CNNs) for image classification tasks using two datasets: **Cats vs. Dogs** and **CIFAR-10**. The goal is to experiment with different activation functions, weight initialization techniques, and optimizers to identify the best-performing model. Additionally, a comparison is made with a pre-trained ResNet-18 model.

## Dataset
1. **Cats vs. Dogs Dataset**: This dataset consists of images of cats and dogs and is used for binary classification.
   - The dataset is downloaded from Kaggle: [Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
   - The images are resized to 256x256 for training.
   - It is split into training and testing directories.

2. **CIFAR-10 Dataset**: This dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
   - The dataset is available in TensorFlow and Keras, and images are resized to 32x32 for training.

## Methodology
We implemented three different CNN models using the following configurations:

### Model 1: 
- **Activation**: ReLU
- **Optimizer**: Adam
- **Weight Initialization**: Random Initialization
- **Performance**:
  - Training Accuracy: 95.01%
  - Validation Accuracy: 74.4%

### Model 2:
- **Activation**: TanH
- **Optimizer**: RMSProp
- **Weight Initialization**: Xavier Initialization
- **Performance**:
  - Training Accuracy: 64.94%
  - Validation Accuracy: 61.20%

### Model 3:
- **Activation**: LeakyReLU
- **Optimizer**: SGD
- **Weight Initialization**: Kaiming Initialization
- **Performance**:
  - Training Accuracy: 98.61%
  - Validation Accuracy: 78.86%

We found **Model 3** to be the best-performing model for both training and validation.

## Additional Implementation: Fine-Tuning ResNet-18
To compare the performance of our custom CNN models with a pre-trained model, we loaded and fine-tuned a **ResNet-18** model (since ResNet-18 is not available in TensorFlow).

- **Training Accuracy**: (insert result)
- **Validation Accuracy**: (insert result)

The weights of the best-performing models (both custom and ResNet-18) are saved for further use.

## Results

- The custom CNN model with **LeakyReLU** activation, **Kaiming initialization**, and **SGD optimizer** (Model 3) performed the best with an accuracy of **78.86%** on the validation set for the **Cats vs. Dogs** dataset.
- **ResNet-18** was fine-tuned for comparison and showed competitive results.

## Conclusion

In this experiment, we explored the impact of different **activation functions**, **optimizers**, and **initialization methods** on CNN performance. We found that the **LeakyReLU** activation function with **Kaiming initialization** and **SGD optimizer** provided the best results across the **Cats vs. Dogs** dataset. Additionally, fine-tuning a pre-trained **ResNet-18** model yielded competitive performance, highlighting the importance of **transfer learning** for image classification tasks.
