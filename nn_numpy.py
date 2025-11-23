import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        #initialize weights with random values
        self.w1 = np.random.randn(input_size, hidden_size) #2x2
        self.b1 = np.zeros((1,hidden_size))

        self.w2 = np.random.randn(hidden_size,output_size) #2x1
        self.b2 = np.zeros((1, output_size))

        self.lr = learning_rate

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x* (1-x)
    