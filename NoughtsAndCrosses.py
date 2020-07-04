import numpy as np

class naughts_crosses:
    def __init__(self, player1, player2, board=1, n = 9):
        self.player1 = player1
        self.player2 = player2
        self.board = np.ones(n).reshape((int(np.sqrt(n))), int(np.sqrt(n)))


    def player1go(self):
        print ("player 1, your turn...")
        xco_ord = int(input("choose your x co-ordinates (0/1/2)"))
        yco_ord = int(input("choose your y co-ordinates (0/1/2)"))
        if self.board[yco_ord, xco_ord] == 1:
            self.board[yco_ord, xco_ord] = '0'
            print (self.board)
        else:
            print ("that tile has already been taken, stop cheating!")
            Game.player1go()




    def player2go(self):
        print ("player 2, your turn...")
        xco_ord = int(input("choose your x co-ordinates (0/1/2)"))
        yco_ord = int(input("choose your y co-ordinates (0/1/2)"))
        if self.board[yco_ord, xco_ord] == 1:
            self.board[yco_ord, xco_ord] = '4'
            print (self.board)
        else:
            print ("that tile has already been taken, stop cheating!")
            Game.player2go()




    def playgame(self):
        win_condition = 0
        player1turn = 0
        player2turn = 0

        win_conds = [self.board[0,:], self.board[1,:], self.board[2,:], self.board[:,0], self.board[:,1], self.board[:,2]]
        win_template = np.zeros(3)
        win_template2 = np.ones(3)*4
        while win_condition == 0:
            if player1turn%2 == 0:
                Game.player1go()
                for win in win_conds:
                    if np.array_equal(win_template,win) == True:
                        print (f"congratulations {self.player1}!")
                        win_condition = 1
                diag1 = (np.array([self.board[2,0], self.board[1,1], self.board[0,2]]))
                diag2 = (np.array([self.board[0,0],self.board[1,1], self.board[2,2]]))
                if np.array_equal(win_template, diag1) == True:
                        print (f"congratulations {self.player1}!")
                        win_condition = 1
                if np.array_equal(win_template, diag2) == True:
                        print (f"congratulations {self.player1}!")
                        win_condition = 1
                player1turn += 1
            else:
                Game.player2go()
                for win in win_conds:
                    if np.array_equal(win_template2,win) == True:
                        print (f"congratulations {self.player2}!")
                        win_condition = 1
                diag1 = (np.array([self.board[2,0], self.board[1,1], self.board[0,2]]))
                diag2 = (np.array([self.board[0,0],self.board[1,1], self.board[2,2]]))
                if np.array_equal(win_template2, diag1) == True:
                        print (f"congratulations {self.player2}!")
                        win_condition = 1
                if np.array_equal(win_template2, diag2) == True:
                        print (f"congratulations {self.player2}!")
                        win_condition = 1
                player1turn += 1



Game = naughts_crosses('Rory', 'Louisa')

Game.playgame()
