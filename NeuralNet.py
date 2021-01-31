import numpy as np
#import random
## neural network

def relu(input_layer):
    return np.maximum(0, input_layer)

def softmax(input_layer):
    calc_exp = np.exp(input_layer - np.max(input_layer, axis=0, keepdims=True)) #axis = 1 ?
    return calc_exp / np.sum(calc_exp, axis=0, keepdims=True) # axis = 1 ?

class NeuralNetwork:
    def __init__(self):
        self.bias = np.random.randint(1,3)
        self.weights_input_to_hidden = np.random.rand(54,45)
        self.weights_hidden_to_output = np.random.rand(45,27)
        self.desired_output_joined = np.array([111, 112, 113, 121, 122, 123, 131, 132, 133,
                                       211, 212, 213, 221, 222, 223, 231, 232, 233,
                                       311, 312, 313, 321, 322, 323, 331, 332, 333])
        self.desired_output = [[1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 2, 1], [1, 2, 2], [1, 2, 3], [1, 3, 1],
                               [1, 3, 2], [1, 3, 3], [2, 1, 1], [2, 1, 2], [2, 1, 3], [2, 2, 1], [2, 2, 2],
                               [2, 2, 3], [2, 3, 1], [2, 3, 2], [2, 3, 3], [3, 1, 1], [3, 1, 2], [3, 1, 3],
                               [3, 2, 1], [3, 2, 2], [3, 2, 3], [3, 3, 1], [3, 3, 2], [3, 3, 3]]
        self.ga = GeneticAlgorithm(self.desired_output, self.bias, self.weights_input_to_hidden, self.weights_hidden_to_output)


    def feedforward(self, input_layer):
        ## relu for input to hidden
        self.layer_input_to_hidden = relu(np.dot(input_layer, self.weights_input_to_hidden) + self.bias)
        ## softmax for hidden to output
        #print(self.layer_input_to_hidden)
        #print(len(self.layer_input_to_hidden))
        self.layer_hidden_to_output = softmax(np.dot(self.layer_input_to_hidden, self.weights_hidden_to_output) + self.bias)
        #print(self.layer_hidden_to_output)
        #print(len(self.layer_hidden_to_output))
        return self.layer_hidden_to_output

    def train(self, input_layer, myScore, oppScore):
        feedforward_res = self.feedforward(input_layer)
        get_max_weight = np.argmax(feedforward_res,axis=0) ## get index of maximum output neron weight
        get_max_weight_val = int(self.desired_output_joined[[get_max_weight]]) ## get value from index above
        res_clear_move = [int(z) for z in str(get_max_weight_val)] ## Format get_max_weight_val as int list valid for game input
        res = [v-1 for v in res_clear_move] ## Minus 1 from move result for valid input
        #self.ga.fitness(feedforward_res, myScore, oppScore, res)
        return res

    def get_weights_input_to_hidden(self):
        return self.weights_input_to_hidden

    def get_weights_hidden_to_output(self):
        return self.weights_hidden_to_output

    def get_bias(self):
        return self.bias

## genetic algorithm

class GeneticAlgorithm:
    def __init__(self, desired_output, bias, weights_input_to_hidden, weights_hidden_to_output):
        self.desired_output = desired_output
        self.bias = bias
        self.weights_input_to_hidden = weights_input_to_hidden
        self.weights_hidden_to_output = weights_hidden_to_output
        self.individual_score = np.zeros(10)
        #self.solution_representation()

    def solution_representation(self):
        #print(*self.desired_output, sep="\n") ## Possible game move solutions are hard coded into self.desired_output
        #self.fitness()
        pass

    def fitness(self, feedforward_res, myScore, oppScore, res):
        score_difference = myScore - oppScore
        if myScore > oppScore: ## if NNPlayer is performing well
            pass

        else: ## NNPlayer is performing well make minor adjustments
            ## Alter fitness by small margin
            pass

    def selection(self):
        pass

    def crossover(self): # reproduction
        pass

    def mutation(self): # redproduction
        pass

    '''
    def find_new_solution(self):
        pass
    '''