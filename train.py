import numpy as np
from nn_numpy import NeuralNetwork
from utils import create_xor_data


def main():
    X, y = create_xor_data()

    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=1.0, seed=1)

    epochs = 20000
    loss_list = []

    for epoch in range(epochs):
        output = nn.forward(X)
        nn.backward(X, y)

        loss = nn.compute_loss(y, output)
        loss_list.append(loss)

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    print("\nTesting predictions:")
    for x, t in zip(X, y):
        pred, prob = nn.predict(np.array([x]))
        print(f"{x} → Pred: {pred[0][0]}, Prob: {prob[0][0]:.4f}, Target: {t[0]}")


if __name__ == "__main__":
    main()
