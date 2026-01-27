# Planar_Data_Classification_with_one_hidden_layer

## Neural Network from Scratch on Planar Datasets

This repository implements a two-layer neural network from scratch using NumPy to classify simple 2D planar datasets. The project focuses on learning non-linear decision boundaries for datasets such as spirals, moons, circles, and blobs, without using high-level machine learning frameworks.

It is designed for educational purposes to build an intuitive understanding of forward propagation, backpropagation, and gradient-based optimization in shallow neural networks.

---

## Features

- Two-layer neural network implemented from scratch using NumPy  
- Manual implementation of forward and backward propagation  
- Binary cross-entropy cost computation  
- Gradient descent–based parameter updates  
- Visualization of 2D datasets and learned decision boundaries  
- Comparison with logistic regression baseline  
- Experiments with varying hidden layer sizes  
- Support for multiple synthetic datasets  

---

## Dataset

The neural network is trained on synthetic 2D datasets generated using `sklearn.datasets`, including:

- Planar dataset  
- Noisy circles  
- Noisy moons  
- Gaussian quantiles  
- Random blobs  

All datasets are visualized with color-coded class labels and loaded using utility functions.

---

## Architecture

### Two-Layer Neural Network

- Input layer size: number of input features (2 for 2D datasets)  
- Hidden layer: configurable number of neurons (default = 4)  
- Output layer: binary classification using sigmoid activation  

### Activation Functions

- Hidden layer: `tanh`  
- Output layer: `sigmoid`  

---

## Learning Process

The training pipeline consists of:

- Forward propagation to compute activations  
- Cost computation using binary cross-entropy loss  
- Backward propagation to compute gradients  
- Gradient descent for parameter updates  

Training is performed over a fixed number of iterations with a configurable learning rate. Cost values are logged and used to assess convergence.

---

## Evaluation

Model performance is evaluated using:

- Training accuracy  
- Decision boundary visualization on 2D datasets  
- Comparison with logistic regression to highlight non-linear modeling capability  
- Accuracy and boundary behavior for different hidden layer sizes  

---

## Experiments

The repository allows experimentation with:

- Hidden layer size (`n_h`)  
- Number of training iterations  
- Learning rate  
- Dataset type  

Results are visualized and corresponding accuracies are printed for comparison.

---

## Requirements

The project requires the following Python libraries:

- numpy  
- matplotlib  
- scikit-learn  


