import numpy as np
## neural network

def relu(input_layer):
    return np.maximum(0, input_layer)

def softmax(input_layer):
    calc_exp = np.exp(input_layer - np.max(input_layer, axis=1, keepdims=True))
    return calc_exp / np.sum(calc_exp, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self):
        self.bias = 3
        self.weights_input_to_hidden = np.random.rand(54,45)
        self.weights_hidden_to_output = np.random.rand(45,27)
        self.desired_output = np.array([1])

    def feedforward(self, input_layer):
        ## relu for input to hidden
        self.layer_input_to_hidden = relu(np.dot(input_layer, self.weights_input_to_hidden) + self.bias)
        ## softmax for hidden to output
        self.layer_hidden_to_output = softmax(np.dot(self.weights_input_to_hidden, self.weights_hidden_to_output) + self.bias)
        return self.layer_hidden_to_output

    def train(self, input_layer):
        feedforward_res = self.feedforward(input_layer)
        print(feedforward_res)
        res = GeneticAlgorithm(feedforward_res, self.desired_output)
        return res

## genetic algorithm

class GeneticAlgorithm:
    def __init__(self, feedforward_res, desired_output):
        self.feedforward_res = feedforward_res
        self.desired_output = desired_output

