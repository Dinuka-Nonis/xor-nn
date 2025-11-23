import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.lr = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y):
        m = y.shape[0]

        error_output = self.a2 - y
        d_w2 = self.a1.T @ error_output / m
        d_b2 = np.sum(error_output, axis=0, keepdims=True) / m

        error_hidden = (error_output @ self.w2.T) * self.sigmoid_derivative(self.a1)
        d_w1 = X.T @ error_hidden / m
        d_b1 = np.sum(error_hidden, axis=0, keepdims=True) / m

        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int), probs

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )