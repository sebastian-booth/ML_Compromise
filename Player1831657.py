import CompromiseGame as game
import NeuralNet as nn
import numpy as np

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
        self.games_played = 3
        self.count = 0
        self.player_pop = []
        self.indiv_fitness_score = list(np.zeros(self.pop_size))
        self.fitness_wheel = []
        self.parents = []
        self.crossover_pair = []
        self.offspring = []

    def generate_init_pop(self):
        for x in range(self.pop_size):
            self.player_pop.append(NNPlayer())

    def fitness_through_play(self):
        self.indiv_fitness_score = list(np.zeros(self.pop_size))
        self.count = 0
        print(self.player_pop)
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
            #print(self.count)
            self.count += 1
        print(self.score_list)
        extract_my_score = [x[0] for x in self.score_list]
        print(sorted(extract_my_score))  # games won
        # print(int(np.argmax(extract_my_score)))
        print(sorted(self.indiv_fitness_score))  # total score advantage

    def find_parents(self):
        fitness_sum = int(sum(self.indiv_fitness_score))
        self.fitness_wheel = []
        self.parents = []
        temp_player_pop = self.player_pop[:]
        print(self.player_pop)
        print(len(self.player_pop))
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
        self.offspring = []
        self.crossover_pair = []
        for x in zip(*[iter(self.parents)]*2): # zip from https://stackoverflow.com/a/5389547
            print(x)
            self.crossover_pair.append(x)
        print(self.crossover_pair)
        print(len(self.crossover_pair))
        for y in self.crossover_pair:
            ith_crossover = np.random.randint(1, 53)
            hto_crossover = np.random.randint(1, 44)

            p1_ith = y[0].get_nn().get_weights_input_to_hidden()
            p2_ith = y[1].get_nn().get_weights_input_to_hidden()

            p1_hto = y[0].get_nn().get_weights_hidden_to_output()
            p2_hto = y[1].get_nn().get_weights_hidden_to_output()

            self.offspring.append(NNPlayer())
            c1_ith = np.append(p1_ith[:ith_crossover], p2_ith[ith_crossover:], axis=0)
            c1_hto = np.append(p1_hto[:hto_crossover], p2_hto[hto_crossover:], axis=0)
            self.offspring[-1].get_nn().set_weights_input_to_hidden(c1_ith)
            self.offspring[-1].get_nn().set_weights_hidden_to_output(c1_hto)

            ga.mutation(self.offspring[-1])

            self.offspring.append(NNPlayer())
            c2_ith = np.append(p2_ith[:ith_crossover], p1_ith[ith_crossover:], axis=0)
            c2_hto = np.append(p2_hto[:hto_crossover], p1_hto[hto_crossover:], axis=0)
            self.offspring[-1].get_nn().set_weights_input_to_hidden(c2_ith)
            self.offspring[-1].get_nn().set_weights_hidden_to_output(c2_hto)

            ga.mutation(self.offspring[-1])

        '''
            print("-----------------------------------------------------------")
            print(self.offspring[0].get_nn().get_weights_input_to_hidden())
            print("-----------------------------------------------------------")
            print(self.offspring[1].get_nn().get_weights_input_to_hidden())
            print("-----------------------------------------------------------")
            print(self.offspring[0].get_nn().get_weights_hidden_to_output())
            print("-----------------------------------------------------------")
            print(self.offspring[1].get_nn().get_weights_hidden_to_output())
            print("-----------------------------------------------------------")

            
            print(str(y[0]) + " Parent 1 - i->h")
            print(y[0].get_nn().get_weights_input_to_hidden())
            print(str(y[1]) + " Parent 2 - i->h")
            print(y[1].get_nn().get_weights_input_to_hidden())

            print(str(y[0]) + " Parent 1 - h->o")
            print(y[0].get_nn().get_weights_hidden_to_output())
            print(str(y[1]) + " Parent 2 - h->o")
            print(y[1].get_nn().get_weights_hidden_to_output())
            '''

    def mutation(self, child):
        mutate_chance = np.random.randint(1,4)
        if mutate_chance == 4:
            mutate_ith = child.get_nn().get_weights_input_to_hidden()
            #mutate_ith[np.random.randint(0,len(mutate_ith))]+= np.random.uniform(-1.0, 1.0, 1) # https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
            index = range(len(mutate_ith))
            flip_1, flip_2 = np.random.sample(index, 2)
            mutate_ith[flip_1], mutate_ith[flip_2] = mutate_ith[flip_2], mutate_ith[flip_1]
            child.get_nn().set_weights_input_to_hidden(mutate_ith)

            mutate_hto = child.get_nn().get_weights_hidden_to_output()
            #mutate_hto[np.random.randint(0,len(mutate_hto))]+= np.random.uniform(-1.0, 1.0, 1) # https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
            index = range(len(mutate_hto)) # swap code from https://stackoverflow.com/a/47724064
            flip_1, flip_2 = np.random.sample(index, 2)
            mutate_hto[flip_1], mutate_hto[flip_2] = mutate_hto[flip_2], mutate_hto[flip_1]
            child.get_nn().set_weights_hidden_to_output(mutate_hto)
        else:
            pass

    def selection(self):
        self.player_pop = []
        if len(self.offspring) >= self.pop_size:
            for x in range (self.pop_size):
                self.player_pop.append(self.offspring[x])
        else:
            self.player_pop = self.offspring

    def play_trained_NNP(self):
        pA = self.player_pop[int(np.argmax(self.indiv_fitness_score))]
        pB = game.SmartGreedyPlayer()
        g = game.CompromiseGame(pA, pB, 30, 10)
        games_played = 5
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
        file = open("percentages_pog.txt", "a")
        file.write("{:.0%}".format(score[0] / games_played) + "\n")
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
        fileo = open("percentages_trash.txt", "a")
        fileo.write("{:.0%}".format(score[0] / games_played) + "\n")
        print("--------------------------")


if __name__ == "__main__":
    generation = 1000
    file = open("percentages_pog.txt", "w")
    file.close()
    fileo = open("percentages_trash.txt", "w")
    fileo.close()
    ga = GeneticAlgorithm()
    ga.generate_init_pop()
    for x in range(generation):
        ga.fitness_through_play()
        ga.find_parents()
        ga.crossover()
        ga.selection()
    ga.fitness_through_play()
    for x in range(10):
        ga.play_trained_NNP()
    # play the game for real

# --------------------------------------------------------------------------------------------