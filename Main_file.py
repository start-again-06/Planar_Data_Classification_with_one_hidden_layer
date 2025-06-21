import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import (
    plot_decision_boundary,
    sigmoid,
    load_planar_dataset,
    load_extra_datasets,
)
np.random.seed(1)
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1] 

print("The shape of X is: " + str(shape_X))
print("The shape of Y is: " + str(shape_Y))
print("I have m = %d training examples!" % (m))
clf = sklearn.linear_model.LogisticRegressionCV()
Y_reshaped = Y.ravel()
clf.fit(X.T, Y_reshaped);
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
LR_predictions = clf.predict(X.T)
print(
    "Accuracy of logistic regression: %d "
    % float(
        (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions))
        / float(Y.size)
        * 100
    )
    + "% "
    + "(percentage of correctly labelled datapoints)"
)
def layer_sizes(X, Y):
n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
 return (n_x, n_h, n_y)
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
def initialize_parameters(n_x, n_h, n_y):
np.random.seed(
        2
    ) 
scale_factor = 0.01
    W1 = np.random.randn(n_h, n_x) * scale_factor
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * scale_factor
    b2 = np.zeros((n_y, 1))
assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters
n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
def forward_propagation(X, parameters):
  W1 = parameters.get("W1")
    b1 = parameters.get("b1")
    W2 = parameters.get("W2")
    b2 = parameters.get("b2")
Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
  assert A2.shape == (1, X.shape[1])
cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

print(
    f"Z1: {cache.get("Z1")}\n \
    A1: {cache.get("A1")}\n \
    Z2: {cache.get("Z2")}\n \
    A2: {cache.get("A1")}\n"
)
print(
    np.mean(cache["Z1"]),
    np.mean(cache["A1"]),
    np.mean(cache["Z2"]),
    np.mean(cache["A2"]),
)

def compute_cost(A2, Y, parameters):
m = Y.shape[1]
 logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)
cost = float(np.squeeze(cost))
 assert isinstance(cost, float)

    return cost
A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
def backward_propagation(parameters, cache, X, Y):
 m = X.shape[1] W1 = parameters.get("W1")
    W2 = parameters.get("W2")
A1 = cache.get("A1")
    A2 = cache.get("A2")
dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (
        1 - np.power(A1, 2)
    )  
  dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
 grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))
def update_parameters(parameters, grads, learning_rate):
W1 = parameters.get("W1")
    b1 = parameters.get("b1")
    W2 = parameters.get("W2")
    b2 = parameters.get("b2")
dW1 = grads.get("dW1")
    db1 = grads.get("db1")
    dW2 = grads.get("dW2")
    db2 = grads.get("db2")
W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 1.2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
def nn_model(X, Y, n_h, learning_rate, num_iterations=10000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
for i in range(0, num_iterations):
        
        A2, cache = forward_propagation(X, parameters)
       
        cost = compute_cost(A2, Y, parameters)
     
        grads = backward_propagation(parameters, cache, X, Y)
      
        parameters = update_parameters(parameters, grads, learning_rate)
       
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
 
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(
    X_assess, Y_assess, 4, 1.02, num_iterations=10000, print_cost=True
)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
def predict(parameters, X):
A2, _ = forward_propagation(X, parameters)
    predictions = A2 > 0.5
return predictions
parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))
parameters = nn_model(X, Y, 4, 1.2, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
predictions = predict(parameters, X)
print(
    "Accuracy: %d"
    % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T))
        / float(Y.size)
        * 100
    )
    + "%"
)
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title("Hidden Layer of size %d" % n_h)
    parameters = nn_model(X, Y, n_h, 1.2, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T))
        / float(Y.size)
        * 100
    )
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = (
    load_extra_datasets()
)

datasets = {
    "noisy_circles": noisy_circles,
    "noisy_moons": noisy_moons,
    "blobs": blobs,
    "gaussian_quantiles": gaussian_quantiles,
}
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y % 2
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
