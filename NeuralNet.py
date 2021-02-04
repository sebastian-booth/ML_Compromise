import numpy as np

def relu(input):
    # Return the rectified linear for the hidden layer
    return np.maximum(0, input)

def softmax(input):
    # Return the softmax probabilities for the output layer
    calc_exp = np.exp(input - np.max(input, axis=0, keepdims=True))
    return calc_exp / np.sum(calc_exp, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self):
        # Randomly generate bias between 1 and 3 and layer weights between 0 and 1 in format [54, 45, 45, 27]
        # load desired output
        self.bias = np.random.randint(1,3)
        self.weights_hidden = np.random.rand(54, 45)
        self.weights_output = np.random.rand(45, 27)
        self.desired_output_joined = np.array([111, 112, 113, 121, 122, 123, 131, 132, 133,
                                       211, 212, 213, 221, 222, 223, 231, 232, 233,
                                       311, 312, 313, 321, 322, 323, 331, 332, 333])

    def feedforward(self, input_layer):
        # Load input layer and hidden layer weights as a dot product, adding bias and return relu result
        # Load values returned from relu  and the output layer weights as a dot product, adding bias and return softmax result
        # Return softmax probabilities
        self.layer_hidden = relu(np.dot(input_layer, self.weights_hidden) + self.bias)
        self.layer_output = softmax(np.dot(self.layer_hidden, self.weights_output) + self.bias)
        return self.layer_output

    def get_move(self, input_layer):
        # Run feedforward method, get the index of the maximum output layer weight, get the value of this index and map
        # onto desired output array, reformat the value as an int list and minus 1 from each element to make it a valid game input
        feedforward_res = self.feedforward(input_layer)
        get_max_weight = np.argmax(feedforward_res,axis=0)
        get_max_weight_val = int(self.desired_output_joined[[get_max_weight]])
        res_clear_move = [int(z) for z in str(get_max_weight_val)]
        res = [v-1 for v in res_clear_move]
        return res

    # get and set self variable functions
    def get_weights_hidden(self):
        return self.weights_hidden

    def get_weights_output(self):
        return self.weights_output

    def get_bias(self):
        return self.bias

    def set_weights_hidden(self, set_val):
        self.weights_hidden = set_val

    def set_weights_output(self, set_val):
        self.weights_output = set_val

    def set_bias(self, set_val):
        self.bias = set_val