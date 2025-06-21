# Planar_Data_Classification_with_one_hidden_layer
# 🧠 Neural Network from Scratch on Planar Datasets

This repository implements a **2-layer neural network** from scratch using NumPy to classify data from simple 2D datasets like spirals, moons, circles, and blobs. It includes comparisons with traditional logistic regression, full visualization of decision boundaries, and scalability with different hidden layer sizes.

---

## 📌 Features

- 📊 Visualization of various 2D datasets using Matplotlib
- 🔁 Implementation of:
  - Layer size initialization
  - Parameter initialization
  - Forward propagation
  - Cost computation
  - Backward propagation
  - Parameter updates
- ⚙️ Custom training loop with learning rate and iteration control
- ✅ Prediction and accuracy evaluation
- 📈 Comparison with logistic regression baseline
- 🧪 Experiments with varying hidden layer sizes
- 🌐 Generalization to multiple synthetic datasets

---

## 📁 Dataset Overview

The neural network is trained on 2D synthetic datasets such as:

- Planar dataset
- Noisy circles
- Noisy moons
- Gaussian quantiles
- Random blobs

All datasets are loaded using utility functions and visualized with color-coded class labels.

---

## ⚙️ Model Architecture

### Two-Layer Neural Network

The network consists of:

- **Input layer**: size = number of input features (2 for 2D data)
- **Hidden layer**: customizable number of neurons (default = 4)
- **Output layer**: binary classification using sigmoid activation

**Activations:**
- Hidden layer: `tanh`
- Output layer: `sigmoid`

---

## 🔍 Key Implementation Components

### 1. Layer Size Initialization
Calculates input, hidden, and output layer sizes based on dataset dimensions.

### 2. Parameter Initialization
Weights are initialized using small random numbers, and biases are set to zeros.

### 3. Forward Propagation
Computes intermediate values (`Z1`, `A1`, `Z2`, `A2`) using matrix operations and activation functions.

### 4. Cost Computation
Uses binary cross-entropy to calculate the error between predicted and actual labels.

### 5. Backward Propagation
Computes gradients of the cost with respect to parameters using chain rule and backpropagation logic.

### 6. Parameters Update
Applies gradient descent with a specified learning rate to minimize the cost function.

### 7. Model Training
Combines all the above steps in a loop to train the neural network over a set number of iterations.

### 8. Prediction
Makes class predictions based on the final output of the trained model.

---

## 📈 Evaluation

The model is evaluated using:

- **Training Accuracy**: Measures how well the model fits the training data.
- **Decision Boundary Visualization**: Shows model performance visually for 2D inputs.
- **Comparison with Logistic Regression**: Illustrates the power of non-linear models over linear ones.
- **Experiments with Hidden Layer Sizes**: Evaluates accuracy and decision boundaries for hidden layer sizes ranging from 1 to 50.

---

## 🔁 Experiments and Results

You can experiment with different values of:

- Hidden layer size (`n_h`)
- Number of iterations
- Learning rate
- Dataset type

Visual results are plotted, and corresponding accuracies are printed for comparison.

---

## 🧪 Supported Datasets

This project supports multiple synthetic datasets via `sklearn.datasets`:
- `planar` (default)
- `noisy_moons`
- `noisy_circles`
- `gaussian_quantiles`
- `blobs`

Each dataset has its own structure and challenges, making them ideal for exploring the capacity of neural networks to model non-linear decision boundaries.

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
requirements.txt

txt
Copy
Edit
numpy
matplotlib
scikit-learn
🧰 File Structure
planar_utils.py: Utility functions for data loading and plotting

testCases_v2.py: Test cases for verifying function outputs

nn_model: Main training loop

predict: Final classifier for new data

plot_decision_boundary: Visualizes model decision regions
