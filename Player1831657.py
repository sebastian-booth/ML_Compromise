import CompromiseGame as game
import NeuralNet as nn
import numpy as np
import random

class NNPlayer(game.AbstractPlayer):
    def __init__(self):
        self.player_unique_nn = nn.NeuralNetwork()

    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        myState = np.array(myState).flatten()
        oppState = np.array(oppState).flatten()
        input_layer = list(myState) + list(oppState)
        res = self.player_unique_nn.train(input_layer, myScore, oppScore)
        #res = [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]
        return res

    def get_nn(self):
        return self.player_unique_nn

class GeneticAlgorithm:
    def __init__(self):
        self.score_list = []
        self.pB = game.SmartGreedyPlayer()
        self.pop_size = 100
        self.games_played = 20
        self.count = 0
        self.player_pop = []
        self.indiv_fitness_score = list(np.zeros(self.pop_size))
        self.fitness_wheel = []
        self.parents = []
        self.crossover_pair = []

    def generate_init_pop(self):
        for x in range(self.pop_size):
            # nn_obj = nn.NeuralNetwork()
            self.player_pop.append(NNPlayer())

    def fitness_through_play(self):
        for y in self.player_pop:
            res_total = 0
            pA = y
            g = game.CompromiseGame(pA, self.pB, 30, 10)
            score = [0, 0, 0]
            for i in range(self.games_played):
                g.resetGame()
                res = g.play()
                # print(y.__dict__)
                # input()
                res_diff = res[0] - res[1]  # minus score red (ME) from green (opp)
                if res[0] > res[1]:  # if red greater than green
                    score[0] += 1  # give red point
                    res_total += res_diff
                elif res[1] > res[0]:  # if green greater than red
                    score[2] += 1  # give green point
                else:
                    score[1] += 1  # tie
            # print("red - player A (ME)     green - player B)
            self.score_list.append(score)
            self.indiv_fitness_score[self.count] = res_total
            print(self.count)
            self.count += 1
        print(self.score_list)
        extract_my_score = [x[0] for x in self.score_list]
        print(sorted(extract_my_score))  # games won
        # print(int(np.argmax(extract_my_score)))
        print(sorted(self.indiv_fitness_score))  # total score advantage

    def find_parents(self): # take 2
        fitness_sum = sum(self.indiv_fitness_score)
        temp_player_pop = self.player_pop[:]
        fitness_buildup = 0
        for x in self.indiv_fitness_score:
            self.fitness_wheel.append(x/fitness_sum)
        print(self.fitness_wheel)
        print(len(self.fitness_wheel))
        print(sum(self.fitness_wheel)) # This doesn't always add up to exactly 1 - I dont know why
        for y in range(self.pop_size//2):
            spin_wheel = np.random.random(fitness_sum)[0]
            print(spin_wheel)
            for (z, selected_parent) in enumerate(temp_player_pop):
                fitness_buildup+=self.fitness_wheel[z]
                if fitness_buildup > spin_wheel:
                    self.parents.append(selected_parent)
                    temp_player_pop.pop(z)
                    break
        print(self.parents)
        print(len(self.parents))

    def crossover(self):
        for x in zip(*[iter(self.parents)]*2): # zip from https://stackoverflow.com/a/5389547
            print(x)
            self.crossover_pair.append(x)
        print(self.crossover_pair)
        print(len(self.crossover_pair))
        input()

        '''
        get:
        ?self.bias
        self.weights_input_to_hidden
        self.weights_hidden_to_output
        '''

    def mutation(self):
        mutate_chance = random.randint(1,4)
        if mutate_chance == 4:
            pass
        else:
            pass

    def selection(self):
        pass

    def test_fitness(self):
        pA = self.player_pop[int(np.argmax(self.indiv_fitness_score))]
        pB = game.SmartGreedyPlayer()
        g = game.CompromiseGame(pA, pB, 30, 10)
        games_played = 50
        score = [0, 0, 0]
        for i in range(games_played):
            g.resetGame()
            res = g.play()
            if res[0] > res[1]:  # if red greater than green
                score[0] += 1  # give red point
            elif res[1] > res[0]:  # if green greater than red
                score[2] += 1  # give green point
            else:
                score[1] += 1  # tie
        print("red - player A (ME)     green - player B")
        print("pog")
        print(score)
        print("{:.0%}".format(score[0] / games_played))
        # score_good = score

        pA = NNPlayer()
        g = game.CompromiseGame(pA, pB, 30, 10)
        # curses.wrapper(g.fancyPlay)

        score = [0, 0, 0]
        for i in range(games_played):
            g.resetGame()
            res = g.play()
            if res[0] > res[1]:  # if red greater than green
                score[0] += 1  # give red point
            elif res[1] > res[0]:  # if green greater than red
                score[2] += 1  # give green point
            else:
                score[1] += 1  # tie
        # print("red - player A (ME)     green - player B")
        # print(sorted(extract_my_score)) # games won
        # print(sorted(indiv_score)) # total score advantage
        # print(score_good)
        print("trash")
        print(score)
        print("{:.0%}".format(score[0] / games_played))


if __name__ == "__main__":
    #generation = 1000
    ga = GeneticAlgorithm()
    ga.generate_init_pop()
    #for x in range(generation):
    ga.fitness_through_play()
    ga.find_parents()
    ga.test_fitness()
    ga.crossover()
    # play the game for real

# --------------------------------------------------------------------------------------------