import CompromiseGame as game
import NeuralNet as nn
import numpy as np

class NNPlayer(game.AbstractPlayer):

    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        myState = np.array(myState).flatten()
        oppState = np.array(oppState).flatten()
        input_layer = list(myState) + list(oppState)
        res = nn_obj.train(input_layer, myScore, oppScore, games_run)
        #res = [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]
        return res


if __name__ == "__main__":
    pA = NNPlayer()
    nn_obj = nn.NeuralNetwork()
    games_run = 10
    print(nn_obj.__dict__)
    pB = game.SmartGreedyPlayer()
    g = game.CompromiseGame(pA, pB, 30, 10)
    #curses.wrapper(g.fancyPlay)

    score = [0,0,0]
    for i in range(games_run):
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
