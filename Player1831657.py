import CompromiseGame as game
import NeuralNet as nn
import numpy as np
#import random

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


if __name__ == "__main__":
    score_list = []
    pB = game.SmartGreedyPlayer()
    pop_size = 50
    count = 0
    player_pop = []
    indiv_score = list(np.zeros(pop_size))
    for x in range(pop_size):
        #nn_obj = nn.NeuralNetwork()
        player_pop.append(NNPlayer())
    games_played = 25
    for y in player_pop:
        res_total = 0
        pA = y
        g = game.CompromiseGame(pA, pB, 30, 10)
        score = [0, 0, 0]
        for i in range(games_played):
            g.resetGame()
            res = g.play()
            print(y.__dict__)
            #input()
            res_diff = res[0] - res[1] # minus from my score from opp
            if res[0] > res[1]:  # if green greater than red
                score[0] += 1  # give green point
                res_total += res_diff
            elif res[1] > res[0]:  # if red greater than green
                score[2] += 1  # give red point
            else:
                score[1] += 1  # tie
        print("green - player A (ME)     red - player B")
        score_list.append(score)
        indiv_score[count] = res_total
        count+=1
    print(score_list)
    extract_my_score = [x[0] for x in score_list]
    print(sorted(extract_my_score)) # games won
    #print(int(np.argmax(extract_my_score)))
    print(sorted(indiv_score)) # total score advantage
    pA = player_pop[int(np.argmax(indiv_score))]
    pB = game.SmartGreedyPlayer()
    g = game.CompromiseGame(pA, pB, 30, 10)
    #curses.wrapper(g.fancyPlay)

    score = [0,0,0]
    for i in range(50):
        g.resetGame()
        res = g.play()
        if res[0] > res[1]: # if red greater than green
            score[0] += 1 # give red point
        elif res[1] > res[0]: # if green greater than red
            score[2] += 1 # give green point
        else:
            score[1] += 1 # tie
    print("green - player A (ME)     red - player B")
    print(score)
    score_good = score

    pA = NNPlayer()
    g = game.CompromiseGame(pA, pB, 30, 10)
    #curses.wrapper(g.fancyPlay)

    score = [0,0,0]
    for i in range(50):
        g.resetGame()
        res = g.play()
        if res[0] > res[1]: # if red greater than green
            score[0] += 1 # give red point
        elif res[1] > res[0]: # if green greater than red
            score[2] += 1 # give green point
        else:
            score[1] += 1 # tie
    print("green - player A (ME)     red - player B")
    print(sorted(extract_my_score)) # games won
    print(sorted(indiv_score)) # total score advantage
    print(score_good)
    print(score)