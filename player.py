import CompromiseGame
import random


class AbstractPlayer:
    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        return [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]

    def placePips(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        return [[random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)] for i in range(nPips)]


class SamplePlayer(AbstractPlayer):
    def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
        return [random.randint(0, 2), random.randint(0, 2), random.randint(0, 2)]


'''
class NNPlayer(AbstractPlayer):
        def play(self, myState, oppState, myScore, oppScore, turn, length, nPips):
            my_grid_val = []
            opp_grid_val = []
            res = [0, 0, 0]
            for x in myState:
                print(x)
                flat_x = [val for sublist in x for val in sublist]
                print(sum(flat_x))
                my_grid_val.append((sum(flat_x)))
            print("---------------------------------------")
            for y in oppState:
                print(y)
                flat_y = [val for sublist in y for val in sublist]
                print(sum(flat_y))
                opp_grid_val.append((sum(flat_y)))
            print("my grid " + str(my_grid_val) + " higher is better")  # higher is better
            print("opp grid " + str(opp_grid_val) + " lower is better")  # lower is better
            my_grid_max = my_grid_val.index(max(my_grid_val))
            opp_grid_max = opp_grid_val.index(min(opp_grid_val))
            if my_grid_max == opp_grid_max:
                res[0] = my_grid_max
            else:
                res[0] = opp_grid_max
            print(res)

            print("\n")
            print("\n")
            print("==============================================")
            my_row_val = []
            opp_row_val = []
            for x in myState[0]:
                print(x)
                print(sum(x))
                my_row_val.append((sum(x)))
            print("---------------------------------------")
            for y in oppState[0]:
                print(y)
                print(sum(y))
                opp_row_val.append((sum(y)))
            print("my row " + str(my_row_val) + " higher is better")  # higher is better
            print("opp row " + str(opp_row_val) + " lower is better")  # lower is better
            my_row_max = my_row_val.index(max(my_row_val))
            opp_row_max = opp_row_val.index(min(opp_row_val))
            if my_row_max == opp_row_max:
                res[1] = my_row_max
            else:
                res[1] = opp_row_max
            print(res)

            res[2] = random.randint(0, 2)
            print(myScore)
            print(oppScore)
            print(turn)
            print(length)
            print(nPips)
            print("end")
            # l = input("end ")
            return res'''