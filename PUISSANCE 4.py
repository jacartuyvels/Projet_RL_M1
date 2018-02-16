# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:38:17 2018

@author: Jacques Cartuyvels and Antoine Van Hoof
"""
##################################
### CONNECT 4 - implementation ###
##################################

#TODO
# - interface graphique
# - number of layers/neurones
# - function d'activation (relu, softmax)
# - dropout
# - convalution layer
# - differents optimizer (GradientDescentOptimizer, AdagradOptimizer, or MomentumOptimizer)


import copy as cp
import numpy as np
import random
import tensorflow as tf

#generates new, empty board
def newBoard():
    board = np.full([6, 7], ' . ')
    return board

#displays game board
def drawBoard(board):
    print('')
    print('    | ' + board[0][0] + board[0][1] + board[0][2] + board[0][3] + board[0][4] + board[0][5] + board[0][6] + ' |')
    print('    | ' + board[1][0] + board[1][1] + board[1][2] + board[1][3] + board[1][4] + board[1][5] + board[1][6] + ' |')
    print('    | ' + board[2][0] + board[2][1] + board[2][2] + board[2][3] + board[2][4] + board[2][5] + board[2][6] + ' |')
    print('    | ' + board[3][0] + board[3][1] + board[3][2] + board[3][3] + board[3][4] + board[3][5] + board[3][6] + ' |')
    print('    | ' + board[4][0] + board[4][1] + board[4][2] + board[4][3] + board[4][4] + board[4][5] + board[4][6] + ' |')
    print('    | ' + board[5][0] + board[5][1] + board[5][2] + board[5][3] + board[5][4] + board[5][5] + board[5][6] + ' |')
    #print('     _  _  _  _  _  _  _  _ ')
    print('')
    
class Game:
    #this class runs the games with our latest trained AI
    def __init__(self, ai, opponent, aiFirst, visible, exploration, freqrand = 3):
        self.ai = ai
        #opponents: 'human', 'random', 'ai'
        self.opponent = opponent
        #does the AI get the first move
        self.aiFirst = aiFirst
        #the boolean visible declares whether the game should be printed so you can follow    
        self.visible = visible
        #exploration vs exploitation: enables a random move once in a while to fully explore the solution space
        self.exploration = exploration
        self.timer = 0
        self.X = freqrand
        #the winner of the game will be decided later
        self.winner = None
        #the full game is kept through a list of all the boards, step by step
        self.trace = LinkedList()
        self.playGame()
        
    #this function plays a game until it's over, then updates the winner field
    def playGame(self):
        if self.getAIFirst() == True:
            while True:
                self.aiMove()
                go, w = isGameOver(self.getTrace().getLastBoard())
                if go:
                    self.winner = w
                    break
                self.opponentMove()
                go, w = isGameOver(self.getTrace().getLastBoard())
                if go:
                    self.winner = w
                    break
        elif self.getAIFirst() == False:
            while True:
                self.opponentMove()
                go, w = isGameOver(self.getTrace().getLastBoard())
                if go:
                    self.winner = w
                    break
                self.aiMove()
                go, w = isGameOver(self.getTrace().getLastBoard())
                if go:
                    self.winner = w
                    break
        if self.getVisible():
            print("Game over -" + self.winner + 'wins!')
            
    def aiMove(self):
        #AI always puts x's
        #enable random moves for solution space exploration
        if self.exploration == True & self.timer%self.X == 0:
            self.timer = self.timer + 1
            self.randomPlay(' x ')
        else:
            self.timer = self.timer + 1
            #first get potential moves that the AI can play based on the current board
            options = self.getOptions(self.getTrace().getLastBoard())
            #let the AI score the potential moves 
            scores = np.array([])
            bestMove = 0
            for i in range(0, options.size):                
                #get a board that represents the move
                newBoard = self.move(self.getTrace().getLastBoard(), int(options[i]), ' x ')
                #AI scores the move 
                scores = np.append(scores, self.getAI().score(newBoard))
                if scores[i] > scores[bestMove]:
                    bestMove = int(i) 
            newTrace = self.getTrace()
            #play the best possible move
            newTrace.addBoard(self.move(newTrace.getLastBoard(), int(options[bestMove]), ' x '))
            self.setTrace(newTrace)
            if self.getVisible():
                print('The AI played:')
                drawBoard(newTrace.getLastBoard())
                
    def opponentMove(self):
        #the opponent always plays with ' o '
        if self.opponent == 'human':
            self.humanPlay()
            print('You played:')
            drawBoard(self.getTrace().getLastBoard())   
        elif self.opponent == 'random':
            self.randomPlay(' o ')
            if self.getVisible():
                print('Random played:')
                drawBoard(self.getTrace().getLastBoard())
        elif self.opponent == 'ai':
            #first get potential moves that the AI can play based on the current board
            options = self.getOptions(self.getTrace().getLastBoard())
            #let the AI score the potential moves 
            scores = np.array([])
            bestMove = 0
            for i in range(0, options.size):                
                #get a board that represents the move
                newBoard = self.move(self.getTrace().getLastBoard(), int(options[i]), ' o ')
                #AI scores the move 
                scores = np.append(scores, self.getAI().score(newBoard))
                if scores[i] > scores[bestMove]:
                    bestMove = int(i) 
            newTrace = self.getTrace()
            #play the best possible move
            newTrace.addBoard(self.move(newTrace.getLastBoard(), int(options[bestMove]), ' o '))
            self.setTrace(newTrace)
            if self.getVisible():
                print('The AI played:')
                drawBoard(newTrace.getLastBoard())

    #here you are the one playing
    def humanPlay(self):
        while True:
            a = input('In what column do you want to add a coin (1-7)?')
            a = int(a)
            a -= 1
            if self.getTrace().getLastBoard()[0][a] == ' . ':
                break
        newTrace = self.getTrace()
        newTrace.addBoard(self.move(newTrace.getLastBoard(), int(a), ' o '))
        self.setTrace(newTrace)
                
    #random player, used for training and presentation
    def randomPlay(self, player):
        #get potential moves
        options = self.getOptions(self.getTrace().getLastBoard())  
        newTrace = self.getTrace()
        #choose a random option
        newTrace.addBoard(self.move(newTrace.getLastBoard(), int(options[random.randint(0,options.size-1)]), player))
        self.setTrace(newTrace)
            
        #get different move options based on current board
    def getOptions(self, board):
        options = np.array([])
        for i in range(0,7):   
            if board[0][i] == ' . ':
                options = np.append(options,i)
        return options
    
    #make a move - define current board, column to insert coin, player making the move
    def move(self, board, col, player):
        #you provide the column in which to insert the coin, in the following loop it "falls down"
        newBoard = board
        i = 5
        while True:
            if newBoard[i][col] == ' . ':
                newBoard[i][col] = player
                break
            i -= 1
        return newBoard
            
    #get the latest version of the game trace (succession of boards for all moves played)
    def getTrace(self):
        return cp.deepcopy(self.trace)
    
    #update the game trace
    def setTrace(self, newTrace):
        self.trace = newTrace    
        
    def getAI(self):
        return self.ai
    
    def getAIFirst(self):
        return self.aiFirst
    
    def getVisible(self):
        return self.visible
        
    def getWinner(self):
        return self.winner
    
#auxiliary classes required for keeping the trace of moves
class Node:
    def __init__(self, board):
        self.cargo = board
        self.next = None

    def getBoard(self):
        return self.cargo
    
    def getNext(self):
        return self.next
    
    def hasNext(self):
        if self.next is None:
            return False
        else:
            return True
    
    def cleanNext(self):
        self.next = None
    
    def addNext(self, node):
        if self.hasNext():
            self.getNext().addNext(node)
        else: 
            self.next = node        
            
    def getLast(self):
        if self.hasNext():
            return self.getNext().getLast()
        else:
            return self.getBoard()
        
    def removeLast(self):
        if self.hasNext():
            if self.getNext().hasNext:
                self.getNext().removeLast()
            else:
                self.cleanNext()
    
#this class is used to keep the trace of nodes
class LinkedList:
    def __init__(self):
        self.head = Node(newBoard())
    
    def getHead(self):
        return self.head
        
    def addBoard(self, board):
        self.getHead().addNext(Node(cp.deepcopy(board)))
            
    def getLastBoard(self):
        return cp.deepcopy(self.getHead().getLast())
    
    def removeLastBoard(self):
        self.getHead().removeLast()    

#checks if game is over - ugly, quick, hard coding
def isGameOver(board):
    gameOver = False
    winner = 'unknown'
    
    #can we spot a vertical 4 in a row?
    for i in range(0,7):
        count_x = 0
        count_o = 0
        for j in range(0,6):
            if board[j][i] == ' x ':
                count_x += 1
                count_o = 0 
                if count_x == 4:
                    gameOver = True
                    winner = ' x '
            elif board[j][i] == ' o ':
                count_o += 1
                count_x = 0
                if count_o == 4:
                    gameOver = True
                    winner = ' o '
            else:
                count_o = 0
                count_x = 0
                   
    #can we spot a horizontal 4 in a row?
    for i in range(0,6):
        count_x = 0
        count_o = 0
        for j in range(0,7):
            if board[i][j] == ' x ':
                count_x += 1
                count_o = 0 
                if count_x == 4:
                    gameOver = True
                    winner = ' x '
            elif board[i][j] == ' o ':
                count_o += 1
                count_x = 0
                if count_o == 4:
                    gameOver = True
                    winner = ' o '
            else:
                count_o = 0
                count_x = 0
    
    #can we spot a diagonal 4 in a row?
    #diagonals up right
    #extract diagonals
    diag_board1 = np.full([6, 6], ' . ')
    diag_board1[2][0] = board[0][3]
    diag_board1[3][0] = board[1][2]
    diag_board1[4][0] = board[2][1]
    diag_board1[5][0] = board[3][0]
    
    diag_board1[1][1] = board[0][4]
    diag_board1[2][1] = board[1][3]
    diag_board1[3][1] = board[2][2]
    diag_board1[4][1] = board[3][1]
    diag_board1[5][1] = board[4][0]
    
    diag_board1[0][2] = board[0][5]
    diag_board1[1][2] = board[1][4]
    diag_board1[2][2] = board[2][3]
    diag_board1[3][2] = board[3][2]
    diag_board1[4][2] = board[4][1]
    diag_board1[5][2] = board[5][0]
    
    diag_board1[0][3] = board[0][6]
    diag_board1[1][3] = board[1][5]
    diag_board1[2][3] = board[2][4]
    diag_board1[3][3] = board[3][3]
    diag_board1[4][3] = board[4][2]
    diag_board1[5][3] = board[5][1]
    
    diag_board1[1][4] = board[1][6]
    diag_board1[2][4] = board[2][5]
    diag_board1[3][4] = board[3][4]
    diag_board1[4][4] = board[4][3]
    diag_board1[5][4] = board[5][2]
    
    diag_board1[2][5] = board[2][6]
    diag_board1[3][5] = board[3][5]
    diag_board1[4][5] = board[4][4]
    diag_board1[5][5] = board[5][3]
    
    #check diagonals
    for i in range(0,6):
        count_x = 0
        count_o = 0
        for j in range(0,6):
            if diag_board1[j][i] == ' x ':
                count_x += 1
                count_o = 0 
                if count_x == 4:
                    gameOver = True
                    winner = ' x '
            elif diag_board1[j][i] == ' o ':
                count_o += 1
                count_x = 0
                if count_o == 4:
                    gameOver = True
                    winner = ' o '
            else:
                count_o = 0
                count_x = 0

    #diagonals down left
    #extract diagonals
    diag_board2 = np.full([6, 6], ' . ')
    diag_board2[2][0] = board[0][3]
    diag_board2[3][0] = board[1][4]
    diag_board2[4][0] = board[2][5]
    diag_board2[5][0] = board[3][6]
    
    diag_board2[1][1] = board[0][2]
    diag_board2[2][1] = board[1][3]
    diag_board2[3][1] = board[2][4]
    diag_board2[4][1] = board[3][5]
    diag_board2[5][1] = board[4][6]
    
    diag_board2[0][2] = board[0][1]
    diag_board2[1][2] = board[1][2]
    diag_board2[2][2] = board[2][3]
    diag_board2[3][2] = board[3][4]
    diag_board2[4][2] = board[4][5]
    diag_board2[5][2] = board[5][6]
    
    diag_board2[0][3] = board[0][0]
    diag_board2[1][3] = board[1][1]
    diag_board2[2][3] = board[2][2]
    diag_board2[3][3] = board[3][3]
    diag_board2[4][3] = board[4][4]
    diag_board2[5][3] = board[5][5]
    
    diag_board2[1][4] = board[1][0]
    diag_board2[2][4] = board[2][1]
    diag_board2[3][4] = board[3][2]
    diag_board2[4][4] = board[4][3]
    diag_board2[5][4] = board[5][4]
    
    diag_board2[2][5] = board[2][0]
    diag_board2[3][5] = board[3][1]
    diag_board2[4][5] = board[4][2]
    diag_board2[5][5] = board[5][3]
    
    #check diagonals
    for i in range(0,6):
        count_x = 0
        count_o = 0
        for j in range(0,6):
            if diag_board2[j][i] == ' x ':
                count_x += 1
                count_o = 0 
                if count_x == 4:
                    gameOver = True
                    winner = ' x '
            elif diag_board2[j][i] == ' o ':
                count_o += 1
                count_x = 0
                if count_o == 4:
                    gameOver = True
                    winner = ' o '
            else:
                count_o = 0
                count_x = 0
    
    #check for draws
    checker = 0
    for i in range(0,7):
        if board[0][i] == ' . ':
            checker += 1
    if checker == 0:
        gameOver = True
        winner = 'draw'       
    
    return(gameOver, winner)

########### TO DO ####################

# function which extracts the features of a board
def extract_feat(board):
    x = 0
    '''
    board_array = np.reshape(board,(3,3))   # reshape into grid form for easier checking
    x = np.zeros(7)
    x[0] = 1
    cols = np.transpose(board_array)
    diag = np.row_stack((np.diag(board_array),np.diag(np.fliplr(board_array))))
    final = np.row_stack((board_array,cols,diag))
    for i in range(0,8):
    # adding features to characterize the condition of the board 
        blank = np.count_nonzero(final[i,:] == ' ')
        X_cnt = np.count_nonzero(final[i,:] == 'X')
        O_cnt = np.count_nonzero(final[i,:] == 'O')
        if X_cnt == 2 and blank == 1:
            x[1] += 1
        elif O_cnt == 2 and blank == 1:
            x[2] += 1
        elif X_cnt == 1 and blank == 2:
            x[3] += 1
        elif O_cnt == 1 and blank == 2:
            x[4] += 1
        elif X_cnt == 3:
            x[5] += 1
        elif O_cnt == 3:
            x[6] += 1
            '''
    return(x)
    
def boardchg(board):
    boardin = np.zeros((84,1))
    #reshape to an array of 42 columns, 1 per board entry
    board = np.reshape(board, (1,42))
    #if board entry has ' x ', 1
    xfields = 1*(board == ' x ')
    #if board entry has ' o ', 1
    ofields = 1*(board == ' o ')
    boardin = np.append(xfields, ofields)
    return boardin
    
    
class AI:
#this is where the magic is supposed to happen - initialised for random play
#TODO to be updated for the AI to learn
    def __init__(self, learningRate):
        self.learningRate = learningRate
        
        #build a graph
        self.X = tf.placeholder(tf.float32, [1,84])
        self.Y = tf.placeholder("float")
        weights = np.random.rand(84)
        self.W = tf.Variable(weights)
        self.Y_model = tf.nn.softmax(tf.matmul(self.X,self.W))
        self.error = tf.square(self.Y - self.Y_model)
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.error)
        self.model = tf.global_variables_initializer()
        self.session = tf.Session().run(self.model)
    
    def score(self, board):
        #Run the session
        boardin = boardchg(board)
        boardin = np.reshape(boardin, (84,1))
        ret = self.session.run(self.Y_model, {self.X: boardin})
        print(ret)
        return ret
        #x = extract_feat(board)   # extract the features of this test board
        #return np.dot(x, weights)
        #return random.randint(0,100)
    
    def learn(self, data, Y):
        #train
        '''
        print(data)
        print(Y)
        print(np.shape(data))
        print(np.shape(Y))
        '''
        
        #self.W = self.sess.run(self.W, {self.reward: Y})
        return None

#this method should be updated according with the data needs of the AI
def prepTrainingData(finishedGame, reward):
    gameTrace = finishedGame.getTrace()
    gameData = np.zeros((1, 84))
    rotNode = gameTrace.getHead()
    while True:
        #reshape to an array of 42 columns, 1 per board entry
        board = np.reshape(rotNode.getBoard(), (1,42))
        #if board entry has ' x ', 1
        xfields = 1*(board == ' x ')
        #if board entry has ' o ', 1
        ofields = 1*(board == ' o ')
        gameData = np.vstack([gameData, np.append(xfields, ofields)])
        if rotNode.hasNext():
            rotNode = rotNode.getNext()
        else:
            break
    #transform and delete first row of zeros
    gameData = np.asmatrix(np.delete(gameData, (0), axis=0))
    Y = np.full([84,1], reward/gameData.shape[0])
    
    #####
    print(xfields)
    print(ofields) 
    gameOver, winner = isGameOver(finishedGame.getTrace().getLastBoard())
    print(gameOver)
    print(winner)
    if winner == ' x ':
        Z = np.append(xfields,-ofields)
    else:
        Z = np.append(-xfields,ofields)
    print(Z)
    Z = Z*Y[0]
    print(Z)
    return gameData, Y, Z

#train the AI!

class Experiment:
   def __init__(self,learningRate, aiFirst, nbGames, exploration, reward, penalty):
        
        #set seed for repeatability of experiment
        random.seed(123)
       
        #initiate AI to train
        marty = AI(learningRate)

        #keep count of wins, losses and draws
        wldtrain = np.zeros(3)
        wldtest = np.zeros(3)

        #train 
        i = nbGames
        while (i>0):
            trainGame = Game(marty, "random", aiFirst, False, exploration)
            if trainGame.getWinner() == ' x ':
                wldtrain[0] += 1
            elif trainGame.getWinner() == ' o ':
                wldtrain[1] += 1
            else:
                wldtrain[2] += 1
            trainMatrix, Y = prepTrainingData(trainGame, reward)
            marty.learn(trainMatrix, Y)
            i -= 1
            
            #print(trainMatrix)
            #print(np.shape(Y))
        
        #test
        nbTestGames = 100
        j = nbTestGames
        while (j>0):
            #no exploration here, go for exploitation of gained knowledge
            testGame = Game(marty, "random", aiFirst, False, False)
            if testGame.getWinner() == ' x ':
                wldtest[0] += 1
            elif trainGame.getWinner() == ' o ':
                wldtest[1] += 1
            else:
                wldtest[2] += 1
            j -= 1

        print("***OUTCOME***")
        print()
        print("Training results (#):")
        s = 'Win: ' + str(wldtrain[0]) + '; lose: ' + str(wldtrain[1]) + '; draw: ' + str(wldtrain[2])
        print(s)
        print("Training results (%):")
        s = "Win: " + str(wldtrain[0]/nbGames) + "; lose: " + str(wldtrain[1]/nbGames) + "; draw: " + str(wldtrain[2]/nbGames)
        print(s)
        print('')
        print("Test results (#):")
        s = "Win: " + str(wldtrain[0]) + "; lose: " + str(wldtrain[1]) + "; draw: " + str(wldtrain[2])
        print(s)
        print("Test results (%):")
        s = "Win: " + str(wldtrain[0]/nbTestGames) + "; lose: " + str(wldtrain[1]/nbTestGames) + "; draw: " + str(wldtrain[2]/nbTestGames)
        print(s)    

def present():
    #configure experiment parameters & run experiment
    learningRate = 0.5
    aiFirst = True
    nbGames = 100
    exploration = True
    reward = 1
    penalty = -1
    #print(learningRate)
    Experiment(learningRate, aiFirst, nbGames, exploration, reward, penalty)


##################################    
### CONNECT 4 - enjoy the game ###
##################################

    
present()