import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, seed=None):
       """
       Initialize weights and biases.
    """
       if seed is not None:
        np.random.seed(seed)

    # weights and biases
       self.w1 = np.random.randn(input_size, hidden_size) * 0.1
       self.b1 = np.zeros((1, hidden_size))

       self.w2 = np.random.randn(hidden_size, output_size) * 0.1
       self.b2 = np.zeros((1, output_size))

       self.lr = learning_rate

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x* (1-x)
    
    def forward(self, X):
        """
        Forward pass through the network.
        Input:
            X: numpy array with shape (m, input_size) where m = batch size (e.g., 4 for XOR)
        Returns:
            z1, a1, z2, a2
            - z1: pre-activation of hidden layer (m, hidden_size)
            - a1: activation of hidden layer (m, hidden_size)
            - z2: pre-activation of output layer (m, output_size)
            - a2: activation (prediction probabilities) of output layer (m, output_size)
        Also stores a1 and a2 on self in case later steps want to use them.
        """
        z1 = X @ self.w1 + self.b1

        a1 = self.sigmoid(z1)

        z2 = a1 @ self.w2 + self.b2

        a2 = self.sigmoid(z2)

        self.last_z1 =z1
        self.last_a1 =a1
        self.last_z2 =z2
        self.last_a2 =a2
        
        return z1, a1, z2, a2

    def predict(self, X, threshold=0.5):
        """
        Convenience method: run forward pass and return binary predictions (0/1)
        and the raw probabilities.
        - threshold: value above which we consider the output a '1'
        Returns:
            preds: integer array (m, output_size) with 0/1 predictions
            probs: float array (m, output_size) with probabilities from 0..1
        """
        _, _, _, a2 = self.forward(X)
        preds = (a2 > threshold).astype(int)
        return preds, a2
    def compute_loss(self, y_true, y_pred):
        """
        Compute Binary Cross Entropy loss.
        y_true: shape (m, 1)
        y_pred: shape (m, 1), probabilities from forward()
        """
        eps = 1e-8  # avoid log(0)
        loss = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        return loss


# Quick manual test when file is executed directly
if __name__ == "__main__":
    # small self-test: create network and run forward on XOR inputs
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, seed=1)
    z1, a1, z2, a2 = nn.forward(X)
    print("z1 (hidden pre-activation):\n", z1)
    print("a1 (hidden activation):\n", a1)
    print("z2 (output pre-activation):\n", z2)
    print("a2 (output probabilities):\n", a2)
    preds, probs = nn.predict(X)
    print("binary predictions:\n", preds)
    X = np.array([[0,0], [0,1], [1,0], [1,1]], float)
    y = np.array([[0],[1],[1],[0]], float)

    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, seed=1)
    _, _, _, a2 = nn.forward(X)

    loss = nn.compute_loss(y, a2)
    print("Loss =", loss)