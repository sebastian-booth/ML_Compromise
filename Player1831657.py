import CompromiseGame as game
import random

class NNPlayer(game.AbstractPlayer):
    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        return [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]


if __name__ == "__main__":
    pA = NNPlayer()
    pB = game.RandomPlayer()
    g = game.CompromiseGame(pA, pB, 30, 5)
    #curses.wrapper(g.fancyPlay)

    score = [0,0,0]
    for i in range(1000):
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
