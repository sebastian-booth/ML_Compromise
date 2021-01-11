import numpy as np
## neural network

def relu(input_layer):
    return np.maximum(0, input_layer)

def softmax(input_layer):
    calc_exp = np.exp(input_layer - np.max(input_layer, axis=0, keepdims=True)) #axis = 1 ?
    return calc_exp / np.sum(calc_exp, axis=0, keepdims=True) # axis = 1 ?

class NeuralNetwork:
    def __init__(self):
        self.bias = 1
        self.weights_input_to_hidden = np.random.rand(54,45)
        self.weights_hidden_to_output = np.random.rand(45,27)
        self.desired_output = np.array([111, 112, 113, 121, 122, 123, 131, 132, 133,
                                       211, 212, 213, 221, 222, 223, 231, 232, 233,
                                       311, 312, 313, 321, 322, 323, 331, 332, 333])

    def feedforward(self, input_layer):
        ## relu for input to hidden
        self.layer_input_to_hidden = relu(np.dot(input_layer, self.weights_input_to_hidden) + self.bias)
        ## softmax for hidden to output
        print(self.layer_input_to_hidden)
        print(len(self.layer_input_to_hidden))
        self.layer_hidden_to_output = softmax(np.dot(self.layer_input_to_hidden, self.weights_hidden_to_output) + self.bias)
        print(self.layer_hidden_to_output)
        print(len(self.layer_hidden_to_output))
        #hold = input()
        return self.layer_hidden_to_output

    def train(self, input_layer):
        feedforward_res = self.feedforward(input_layer)
        res = GeneticAlgorithm(feedforward_res, self.desired_output)
        get_max_weight = np.argmax(feedforward_res,axis=0)
        get_max_weight_val = int(self.desired_output[[get_max_weight]])
        res_clear_move = [int(z) for z in str(get_max_weight_val)]
        res = [v-1 for v in res_clear_move]
        return res

## genetic algorithm

class GeneticAlgorithm:
    def __init__(self, feedforward_res, desired_output):
        self.feedforward_res = feedforward_res
        self.desired_output = desired_output

