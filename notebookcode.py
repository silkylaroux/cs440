
# coding: utf-8

# # Reinforcement Learning Solution to the Towers of Hanoi Puzzle

# In[4]:


import neuralnetworks as nnQ
import random
import numpy as np


# In[103]:


from statistics import mean 


# In[303]:


from copy import copy
from copy import deepcopy


# ## Requirements

# First, how should we represent the state of this puzzle?  We need to keep track of which disks are on which pegs. Name the disks 1, 2, and 3, with 1 being the smallest disk and 3 being the largest. The set of disks on a peg can be represented as a list of integers.  Then the state can be a list of three lists.
# 
# For example, the starting state with all disks being on the left peg would be `[[1, 2, 3], [], []]`.  After moving disk 1 to peg 2, we have `[[2, 3], [1], []]`.
# 
# To represent that move we just made, we can use a list of two peg numbers, like `[1, 2]`, representing a move of the top disk on peg 1 to peg 2.

# Now on to some functions. Define at least the following functions. Examples showing required output appear below.
# 
#    - `printState(state)`: prints the state in the form shown below
#    - `validMoves(state)`: returns list of moves that are valid from `state`
#    - `makeMove(state, move)`: returns new (copy of) state after move has been applied.
#    - `trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF)`: train the Q function for number of repetitions, decaying epsilon at start of each repetition. Returns Q and list or array of number of steps to reach goal for each repetition.
#    - `testQ(Q, maxSteps, validMovesF, makeMoveF)`: without updating Q, use Q to find greedy action each step until goal is found. Return path of states.
# 
# A function that you might choose to implement is
# 
#    - `stateMoveTuple(state, move)`: returns tuple of state and move.  
#     
# This is useful for converting state and move to a key to be used for the Q dictionary.
# 
# Show the code and results for testing each function.  Then experiment with various values of `nRepetitions`, `learningRate`, and `epsilonDecayFactor` to find values that work reasonably well, meaning that eventually the minimum solution path of seven steps is found consistently.
# 
# Make a plot of the number of steps in the solution path versus number of repetitions. The plot should clearly show the number of steps in the solution path eventually reaching the minimum of seven steps, though the decrease will not be monotonic.  Also plot a horizontal, dashed line at 7 to show the optimal path length.
# 
# Add markdown cells in which you describe the Q learning algorithm and your implementation of Q learning as applied to the Towers of Hanoi problem.  Use at least 15 sentences, in one or more markdown cells.

# # Functions
# The following functions are used to implement the Q learning algorithm applied to the Towers of Hanoi problem.
# Each function should have a brief description on how what it does and how it relates to the Q learning algorithm.

# The following function is a helper function, which turns state and move into tuples, this is necessary for iterating over them in the Dictionary Q, it will return the "tuplefide" state and move pair.

# In[309]:


def stateMoveTuple(state, move):
    tempTupState = []
    for x in state:
        tempTupState.append(tuple(x))        
    return (tuple(tempTupState),tuple(move))


# The validMoves function takes in a state checks to see what moves can be made from the current state, it checks the tops of each of the columns and sees if this top can move to either of the other two columns. 

# In[13]:


def validMoves(state):
    validMoves = []
    for x in range(1,4):
        for y in range(1,4):
            if len(state[x-1]) > 0:
                topx = state[x-1]
                if len(state[y-1]) !=0:
                    topy = state[y-1]
                if len(state[y-1]) == 0:
                    validMoves.append([x,y])
                elif((topx != topy and topy > topx)):
                    validMoves.append([x,y])
                elif((topx != topy and topy > topx)):
                    validMoves.append([x,y])
                elif((topx != topy and topy > topx)):
                    validMoves.append([x,y])

    return validMoves                


# The printState function simply prints out the state given to it in a format that is nice to read. It gives a visual of the columns and the "rings" on them. 

# In[380]:


def printState(state):
    temp = 0
    col1Len = 0
    col2Len = 0
    col3Len = 0
    big = len(max(state))
    while temp < len(max(state)):
        holder1,holder2,holder3 = " "," "," "
        
        if col1Len< len(state[0]) and col1Len !=big and col1Len == temp:
            holder1 = state[0][temp]
            col1Len = col1Len + 1
        if col2Len < len(state[1]) and col2Len == temp:
            holder2 = state[1][temp]
            col2Len = col2Len + 1
        if col3Len < len(state[2])and col3Len == temp:
            holder3 = state[2][temp]
            col3Len = col3Len + 1
            
        print(holder1,holder2,holder3)
        temp = temp+1
    print("------")


# The makeMove function returns a copy of the given state after it has taken the move which is given to it.

# In[302]:


def makeMove(state, move):
    state2 = deepcopy(state)
    temp = state2[move[0]-1][0]
    state2[move[0]-1].pop(0)
    state2[move[1]-1].insert(0,temp)
    return state2


# The trainQ function is what actually creates and trains the Q dictionary (the State,Action dictionary). It goes through the given amount of reputations, and updates the value of the Q dictionary based on the choice decided from the epsilonFindMoves(greedyEpsilon). It also decays the epsilon given to the epsilonFindMoves function. 

# In[345]:


def trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF, makeMoveF):
    Q = {}
    
    state = [[1,2,3],[],[]]
    epsilon = 1
    step_count = []
    
    for x in range(nRepetitions):
        
        epsilon = epsilonDecayFactor * epsilon
        step = 0
        goal_state = [[],[],[1,2,3]]
        done = False
        
        while done != True:
            
            step = step + 1
            next_move = epsilonFindMoves(Q, state, epsilon, validMovesF)
            #print(state,next_move)
            next_state = makeMove(state,next_move) 
            
            if stateMoveTuple(state,next_move) not in Q:
                Q[stateMoveTuple(state,next_move)] = 0
                
            if next_state == goal_state:
                Q[stateMoveTuple(state,next_move)] = 1
                done = True
                
            if step > 1:
                Q[stateMoveTuple(stateOld,moveOld)] += learningRate *(1+Q[stateMoveTuple(state,next_move)]-
                                                        Q[stateMoveTuple(stateOld,moveOld)])
            stateOld, moveOld = state, next_move
            state = next_state
        state = [[1,2,3],[],[]]
        step_count.append(step)
    return Q ,step_count


# The testQ function takes in the Q dictionary that was trained, and picks the most greedy option towards the goal. I

# In[355]:


def testQ(Q, maxSteps, validMovesF, makeMovesF):
    
    state = [[1,2,3],[],[]]
    epsilon = 0
    path = []
    #path.append(state)
    
    step = 0
    goal_state = [[],[],[1,2,3]]
    done = False

    while done != True:
        step = step+1
        path.append(state)
        next_move = epsilonFindMoves(Q, state, epsilon, validMovesF)

        next_state = makeMove(state,next_move) 
        

        if next_state == goal_state:
            done = True
        if step > maxSteps:
            done = True
        state = next_state
    return path


# The epsilonFindMoves(...) function takes in the 

# In[301]:


def epsilonFindMoves(Q, state, epsilonRate, validMovesF):
    validMoveList = validMoves(state)
    if np.random.uniform()<epsilonRate:
        return validMoveList[np.random.choice(len(validMoveList))] 
    else:
        Qs = np.array([Q.get(stateMoveTuple(state, m), 0) for m in validMoveList])
        return validMoveList[np.argmin(Qs)]


# In[369]:


for x in path:
    printState(x)


# In[370]:


Q, stepsToGoal = trainQ(1000, 0.5, 0.7, validMoves, makeMove)


# In[371]:


mean(stepsToGoal)


# In[372]:


mean(stepsToGoal)


# # Testing

# In[378]:


state = [[1,2,3],[],[]]
move = [1,2]


# In[377]:


#testing stateMoveTuple(state,move)
print(stateMoveTuple(state,move))


# In[376]:


#testing validMoves(state)
print(validMoves(state))


# In[383]:


#testing printState(state)
printState(state)


# In[384]:


#testing makeMove(state,move)
print(makeMove(state,move))


# In[399]:


#testing trainQ(...)
Q, stepsToGoal = trainQ(50, 0.5, 0.7, validMoves, makeMove)
print('State Actions in Q:' ,len(Q))
print("Progression of steps to goal:")
print(np.array(stepsToGoal))
print('Mean of steps:',mean(steps))


# In[410]:


#testing testQ(...)
Q, stepsToGoal = trainQ(100, 0.5, 0.7, validMoves, makeMove)
path = testQ(Q, 20, validMoves, makeMove)
print("Path for trained Q to Goal:")
for s in path:
    printState(s)


# In[403]:


path


# ## Further investigation

# ### Control

# In[445]:


Q, stepsToGoal = trainQ(500, 0.5, 0.7, validMoves, makeMove)
print('500 Repetitions, .5 learning rate, .7 decay rate')
print('State Actions in Q:' ,len(Q))
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))


# ### Repetitions

# In[436]:


Q, stepsToGoal = trainQ(5, 0.5, 0.7, validMoves, makeMove)
print('5 Repetitions')
print('State Actions in Q:' ,len(Q))
print("Progression of steps to goal:")
print(np.array(stepsToGoal))
print('Mean of steps:',mean(stepsToGoal))


# In[437]:


Q, stepsToGoal = trainQ(500, 0.5, 0.7, validMoves, makeMove)
print('500 Repetitions')
print('State Actions in Q:' ,len(Q))
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))


# In[438]:


Q, stepsToGoal = trainQ(10000, 0.5, 0.7, validMoves, makeMove)
print('10000 Repetitions')
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))


# ### Learning Rate

# In[439]:


Q, stepsToGoal = trainQ(500, .99 , 0.7, validMoves, makeMove)
print('500 Repetitions, .99 learning rate')
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))
Q, stepsToGoal = trainQ(500, .01 , 0.7, validMoves, makeMove)
print('500 Repetitions, .01 learning rate')
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))


# ### Epsilon decay rate

# In[440]:


Q, stepsToGoal = trainQ(500, 0.5, 0.99, validMoves, makeMove)
print('500 Repetitions, .99 decay')
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))
Q, stepsToGoal = trainQ(500, 0.5, 0.01, validMoves, makeMove)
print('500 Repetitions, .01 decay')
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))


# ### Best of 3 options

# In[442]:


Q, stepsToGoal = trainQ(10000, 0.99, 0.01, validMoves, makeMove)
print('10000 Repetitions, .99 learning rate, .01 decay')
print("Progression of steps to goal:")
print(np.array(stepsToGoal[:100]))
print('Mean of steps:',mean(stepsToGoal))


# # Discussion of Results

# There were some very interesting things that I found when testing the trainQ function with different inputs. When testing with different repetitions it was very clear to see that there is a clear benefit to doing more repetitions. Doing only 5 repetitions meant that it took many more than 7 steps to get to the goal, and doing 500 repetitions always reached 7 steps. It didn't seem however that doing an excessive amount of repetitions made it quicker to get to 7 steps. 
# 
# I also tested with differing learning rates both .99 and .01. This did seem to make a big impact on the "learning". The .99 learning rate was able to very quickly lower it down to only 7 moves, it was faster than the control. This was completely the opposite of the .01 learning rate, which only a couple times randomly got 7 moves.
# 
# I then tested different epsilon decay rates .99 and .01. This did seem to make an impact the decay rate of .01 did reach doing 7 moves at about the same rate as the control, whereas .99 took a while to reach 8 moves. Changing decay rate didn't seem to have as much a positive impact as it would have a negative one when compared to the control.
# 
# After doing this, I took the best input option from all the different testings, and it did significantly better than the control. The control had an average of 8.26 and took about 25 repetitions to reach a constant of 7 moves. This is much less effective when compared to the best of all the options I tested, that one had an average of 7.033 moves, and took only 12 to reach a constant of 7 moves. 

# ## Grading

# Download and extract `A4grader.py` from [A4grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A4grader.tar).

# In[348]:


get_ipython().magic('run -i A4grader.py')


# ## Extra Credit

# Modify your code to solve the Towers of Hanoi puzzle with 4 disks instead of 3.  Name your functions
# 
#     - printState_4disk
#     - validMoves_4disk
#     - makeMove_4disk
# 
# Find values for number of repetitions, learning rate, and epsilon decay factor for which trainQ learns a Q function that testQ can use to find the shortest solution path.  Include the output from the successful calls to trainQ and testQ.
