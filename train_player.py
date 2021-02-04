import CompromiseGame as game
import NeuralNet as nn
import numpy as np

class NNPlayer(game.AbstractPlayer):
    def __init__(self):
        # Initialise player neural network
        self.player_unique_nn = nn.NeuralNetwork()

    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        # Setup nn input with game state, pass to nn to retrieve move
        myState = np.array(myState).flatten()
        oppState = np.array(oppState).flatten()
        input_layer = list(myState) + list(oppState)
        res = self.player_unique_nn.get_move(input_layer)
        return res

    def get_nn(self):
        return self.player_unique_nn

class GeneticAlgorithm:
    def __init__(self):
        self.pB = game.SmartGreedyPlayer()
        self.pop_size = 100
        self.games_played = 10
        self.count = 0
        self.player_pop = []
        self.indiv_fitness_score = list(np.zeros(self.pop_size))
        self.fitness_wheel = []
        self.parents = []
        self.crossover_pair = []
        self.offspring = []

    def generate_init_pop(self):
        # Setup initial population
        for x in range(self.pop_size):
            self.player_pop.append(NNPlayer())
        print("Initial population: " + str(len(self.player_pop)))
        print(self.player_pop)
        print("")

    def fitness_through_play(self):
        # Iterate through player population where each individual will play SmartGreedyPlayer for the
        # number in games_played. Gauge individual fitness by adding the score difference for each game between the training player
        # and the opponent. The individual score list will have a matching index to its individual in the population list
        self.indiv_fitness_score = list(np.zeros(self.pop_size))
        for (count, y) in enumerate(self.player_pop):
            res_total = 0
            pA = y
            g = game.CompromiseGame(pA, self.pB, 30, 10)
            score = [0, 0, 0]
            for i in range(self.games_played):
                g.resetGame()
                res = g.play()
                res_diff = res[0] - res[1]  # minus score red (ME) from green (opp)
                if res[0] > res[1]:  # if red greater than green
                    score[0] += 1  # give red point
                    res_total += res_diff
                elif res[1] > res[0]:  # if green greater than red
                    score[2] += 1  # give green point
                else:
                    score[1] += 1  # tie
            self.indiv_fitness_score[count] = res_total
            self.count += 1
        print("Player population: " + str(len(self.player_pop)))
        print(self.player_pop)
        print("Sorted player fitness score: " + str(sorted(self.indiv_fitness_score)))  # total score advantage
        print("")

    def find_parents(self):
        # Roulette wheel selection (fitness proportionate) - calculate fitness sum and append a probability value
        # to a list by dividing the score of an individual by the total fitness sum. fitness_wheel denotes a list of
        # probabilities that a individual will be picked as a parent. Generate parents for half of the population
        # by randomly appending a player to a parents list with the probability of the fitness wheel.
        fitness_sum = int(sum(self.indiv_fitness_score))
        self.fitness_wheel = []
        self.parents = []
        temp_player_pop = self.player_pop[:]
        for x in self.indiv_fitness_score:
            self.fitness_wheel.append(x/fitness_sum)
        print("Fitness wheel probability: " + str(self.fitness_wheel))
        for x in range(self.pop_size//2):
            self.parents.append(np.random.choice(temp_player_pop, p=self.fitness_wheel))
        print("Selected Parents: " + str(len(self.parents)))
        print(self.parents)
        print("")

    def crossover(self):
        # Crossover parents to create offspring - split parents into sublists of two individuals
        # for each parent pair append two offspring. Randomly generate the number of multi-point crossovers
        # select random crossover point index in hidden and output layer weights
        # Get weights for both parents for both hidden and output layer weights
        # Swap weights between parents using a slice index and append to an temporary offspring weight array
        # Update the weights of the new offspring player
        # Do the last two twice with each offspring and using the last and second to last index in offspring list
        # Run each offspring through the mutation method
        # h = hidden layer / o = output layer
        self.offspring = []
        self.crossover_pair = []
        for x in zip(*[iter(self.parents)]*2): # zip from https://stackoverflow.com/a/5389547
            self.crossover_pair.append(x)
        print("Crossover Pair Size: " + str(len(self.crossover_pair)))
        print(self.crossover_pair)
        print("")
        for y in self.crossover_pair:
            self.offspring.append(NNPlayer())
            self.offspring.append(NNPlayer())
            for x in range(np.random.randint(10, 20)):
                h_crossover = np.random.randint(1, 53)
                o_crossover = np.random.randint(1, 44)
                p1_h = y[0].get_nn().get_weights_hidden()
                p2_h = y[1].get_nn().get_weights_hidden()

                p1_o = y[0].get_nn().get_weights_output()
                p2_o = y[1].get_nn().get_weights_output()

                c1_h = np.append(p1_h[:h_crossover], p2_h[h_crossover:], axis=0)
                c1_o = np.append(p1_o[:o_crossover], p2_o[o_crossover:], axis=0)
                self.offspring[-1].get_nn().set_weights_hidden(c1_h)
                self.offspring[-1].get_nn().set_weights_output(c1_o)

                c2_h = np.append(p2_h[:h_crossover], p1_h[h_crossover:], axis=0)
                c2_o = np.append(p2_o[:o_crossover], p1_o[o_crossover:], axis=0)
                self.offspring[-2].get_nn().set_weights_hidden(c2_h)
                self.offspring[-2].get_nn().set_weights_output(c2_o)

            ga.mutation(self.offspring[-1])
            ga.mutation(self.offspring[-2])

    def mutation(self, child):
        # Only mutate 25% of the time, if true get the hidden weights (and its indexes) of the child, get two random
        # indices of weights and swap the values by redeclaring variables with flipped values, set the new weights
        # do the same with the output weights
        mutate_chance = np.random.randint(1,4)
        if mutate_chance == 4:
            mutate_h = child.get_nn().get_weights_hidden()
            index = range(len(mutate_h)) # swap code from https://stackoverflow.com/a/47724064
            flip_1, flip_2 = np.random.sample(index, 2)
            mutate_h[flip_1], mutate_h[flip_2] = mutate_h[flip_2], mutate_h[flip_1]
            child.get_nn().set_weights_hidden(mutate_h)

            mutate_o = child.get_nn().get_weights_output()
            index = range(len(mutate_o)) # swap code from https://stackoverflow.com/a/47724064
            flip_1, flip_2 = np.random.sample(index, 2)
            mutate_o[flip_1], mutate_o[flip_2] = mutate_o[flip_2], mutate_o[flip_1]
            child.get_nn().set_weights_output(mutate_o)
        else:
            pass

    def selection(self):
        # Copy player fitness score into temp variable, if the offspring generated is less then half of the static
        # population size then replace the current worst fitness players and replace with offspring.
        # if the number of offspring is greater than or equal to half the population size then replace half of the
        # current worst players with offspring.
        temp_player_fitness = np.array(self.indiv_fitness_score[:])
        if len(self.offspring) < self.pop_size//2:
            for x in temp_player_fitness.argsort()[:len(self.offspring)]:
                self.player_pop.pop(x)
                self.player_pop.append(self.offspring)
        else:
            for x in temp_player_fitness.argsort()[:self.pop_size//2]:
                self.player_pop.pop(x)
                self.player_pop.append(self.offspring[np.random.randint(0, len(self.offspring))])
        print("New population size: " + str(len(self.player_pop)))
        print(self.player_pop)
        print("")

    def play_trained_NNP(self):
        # Set pA as the player in the trained population with the highest fitness score and play x games against SmartGreedyPlayer
        # Record the results as a percentage against the games played
        # Do the same but set pA to the player in the trained population with the lowest fitness score
        # Return the weights and biases of the best player
        pA = self.player_pop[int(np.argmax(self.indiv_fitness_score))]
        pB = game.SmartGreedyPlayer()
        g = game.CompromiseGame(pA, pB, 30, 10)
        games_played = 1000
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
        print("good player")
        print(score)
        print("{:.0%}".format(score[0] / games_played))
        file = open("misc/percentages_good_player.txt", "a")
        file.write("{:.0%}".format(score[0] / games_played) + "\n")

        pA = self.player_pop[int(np.argmin(self.indiv_fitness_score))]
        g = game.CompromiseGame(pA, pB, 30, 10)

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
        print("bad player")
        print(score)
        print("{:.0%}".format(score[0] / games_played))
        fileo = open("misc/percentages_bad_player.txt", "a")
        fileo.write("{:.0%}".format(score[0] / games_played) + "\n")
        print("--------------------------")
        #print(self.player_pop[int(np.argmax(self.indiv_fitness_score))].get_nn().get_weights_hidden())
        #print(self.player_pop[int(np.argmax(self.indiv_fitness_score))].get_nn().get_weights_output())
        #print(self.player_pop[int(np.argmax(self.indiv_fitness_score))].get_nn().get_bias())

if __name__ == "__main__":
    # Run through the genetic algorithm for the set generation number then recalcluate fitness and play x number
    # of games with the trained population
    generation = 1000
    file = open("misc/percentages_good_player.txt", "w")
    file.close()
    fileo = open("misc/percentages_bad_player.txt", "w")
    fileo.close()
    ga = GeneticAlgorithm()
    ga.generate_init_pop()
    for x in range(generation):
        print("Generation: " + str(x))
        ga.fitness_through_play()
        ga.find_parents()
        ga.crossover()
        ga.selection()
        print("")
    ga.fitness_through_play()
    for x in range(5):
        ga.play_trained_NNP()