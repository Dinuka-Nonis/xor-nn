# XOR Neural Network (NumPy From Scratch)

This project demonstrates how to build, train, and understand a tiny neural network *entirely from scratch* using only **NumPy**. It learns the classic XOR logic function — a minimal but powerful example of why neural networks need non‑linearity.

---

## XOR Function

The network learns to reproduce the behavior of XOR:

| x1 | x2 | XOR |
| -- | -- | --- |
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

---

## Project Structure

```
xor-nn/
│── nn_numpy.py   # Neural network implementation (weights, activations)
│── train.py      # Training loop
│── utils.py      # Optional helper functions
└── README.md     # Project documentation
```

---

## Features

* Neural network coded manually with **pure NumPy**
* Manual **weight and bias initialization**
* **Sigmoid** activation function
* Forward pass and backward pass implemented by hand
* Custom **gradient descent** update step
* Trains successfully to model the XOR function

---

## Learning Goals

This project is meant to teach the core mechanics of neural networks, including:

* How weights and biases behave
* How activation functions affect learning
* How backpropagation computes gradients
* How gradient descent updates parameters
* Why XOR requires a non‑linear model

---

## How to Run

1. Install dependencies:

```bash
pip install numpy
```

2. Train the model:

```bash
python train.py
```

3. Observe the loss decreasing as the network learns XOR.

---

## How It Works

* **Input Layer:** 2 inputs (x1, x2)
* **Hidden Layer:** 2 neurons with sigmoid
* **Output Layer:** 1 sigmoid neuron
* **Loss:** Mean Squared Error (MSE)
* **Optimizer:** Manual gradient descent

The forward and backward passes are computed step‑by‑step, allowing you to understand exactly how each weight influences the outcome.

---

## Why XOR?

XOR cannot be solved with a linear classifier.
A neural network must learn a non‑linear decision boundary — making XOR the perfect beginner challenge.

