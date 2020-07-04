import numpy as np

class Connect4():
    def __init__(self, player1, player2, width = 5, height = 6):
        self.player1 = player1
        self.player2 = player2
        self.board = np.zeros(width*height).reshape(height,width )
        self.width = width  - 1
        self.height = height - 1
        self.counter = [0] * width
        self.gamewon1 = False
        self.gamewon2 = False
        print (self.counter)
        print (self.board)



    def player1turn(self):
        dict = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7, 'i' : 8, 'j' : 9, 'k' : 10}
        choices = (list(dict.keys())[:self.width+1])
        x = input(f"Player one, choose your position {choices}")
        self.board[self.height - self.counter[dict[x]]][dict[x]] = 1
        self.counter[dict[x]] += 1
        #print (self.counter)
        print (self.board)



    def player2turn(self):
        dict = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7, 'i' : 8, 'j' : 9, 'k' : 10}
        choices = (list(dict.keys())[:self.width+1])
        x = input(f"Player two, choose your position({choices}")
        self.board[self.height - self.counter[dict[x]]][dict[x]] = 2
        self.counter[dict[x]] += 1
        print (self.board)

    def play(self):
        turncount = 0
        maxturns = (self.width+1)*(self.height+1)
        while self.gamewon1 == False and self.gamewon2 == False and turncount < maxturns:
            if turncount % 2 == 0:
                game.player1turn()
                game.wincheck1()
            else:
                game.player2turn()
                game.wincheck2()
            turncount+=1
        if self.gamewon1 == True:
            print (f"winner winner chicken dinner, congratulations {self.player1}")
        if self.gamewon2 == True:
            print (f"winner winner chicken dinner, congratulations {self.player2}")

    def wincheck1(self):
        lst = []
        for row in range(self.height+1):
            for i in range(self.width-2):
                if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row,i+1]):
                    if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row,i+2]):
                        if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row,i+3]):
                            lst.append([row,i])

        for row in range(self.height-2):
            for i in range(self.width+1):
                if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+1,i]):
                    if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+2,i]):
                        if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+3,i]):
                            lst.append([row,i])

        for row in range(self.height-2):
            for i in range(self.width-2):
                if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+1,i+1]):
                    if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+2,i+2]):
                        if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+3,i+3]):
                            lst.append([row,i])

        for row in range(self.height-2):
            for i in range(self.width-2,self.width+1):
                if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+1,i-1]):
                    if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+2,i-2]):
                        if (self.board[row,i]) == 1 and (self.board[row,i] == self.board[row+3,i-3]):
                            lst.append([row,i])
        if len(lst) > 0:
            self.gamewon1 = True


    def wincheck2(self):
        lst = []
        for row in range(self.height+1):
            for i in range(self.width-2):
                if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row,i+1]):
                    if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row,i+2]):
                        if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row,i+3]):
                            lst.append([row,i])

        for row in range(self.height-2):
            for i in range(self.width+1):
                if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+1,i]):
                    if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+2,i]):
                        if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+3,i]):
                            lst.append([row,i])

        for row in range(self.height-2):
            for i in range(self.width-2):
                if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+1,i+1]):
                    if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+2,i+2]):
                        if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+3,i+3]):
                            lst.append([row,i])

        for row in range(self.height-2):
            for i in range(self.width-2,self.width+1):
                if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+1,i-1]):
                    if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+2,i-2]):
                        if (self.board[row,i]) == 2 and (self.board[row,i] == self.board[row+3,i-3]):
                            lst.append([row,i])
        if len(lst) > 0:
            self.gamewon2 = True






game = Connect4('teamEugene', 'James')
game.play()
