
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


# In[5]:


nnetQ = nnQ.NeuralNetwork(2, 3, 1)


# For this assignment, you will use reinforcement learning to solve the [Towers of Hanoi](https://en.wikipedia.org/wiki/Tower_of_Hanoi) puzzle.  
# 
# To accomplish this, you must modify the code discussed in lecture for learning to play Tic-Tac-Toe.  Modify the code  so that it learns to solve the three-disk, three-peg
# Towers of Hanoi Puzzle.  In some ways, this will be simpler than the
# Tic-Tac-Toe code.  
# 
# Steps required to do this include the following:
# 
#   - Represent the state, and use it as a tuple as a key to the Q dictionary.
#   - Make sure only valid moves are tried from each state.
#   - Assign reinforcement of $1$ to each move, even for the move that results in the goal state.
# 
# Make a plot of the number of steps required to reach the goal for each
# trial.  Each trial starts from the same initial state.  Decay epsilon
# as in the Tic-Tac-Toe code.

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

# In[309]:


def stateMoveTuple(state, move):
    tempTupState = []
    for x in state:
        tempTupState.append(tuple(x))        
    return (tuple(tempTupState),tuple(move))


# In[300]:


print(stateMoveTuple([[2,3], [1], []], [1 ,2]))


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


# In[327]:


def testQ(Q, maxSteps, validMovesF, makeMovesF):
    
    state = [[1,2,3],[],[]]
    epsilon = 0
    path = []
    
    
    step = 0
    goal_state = [[],[],[1,2,3]]
    done = False

    while done != True:
        step = step+1
        next_move = epsilonFindMoves(Q, state, epsilon, validMovesF)

        next_state = makeMove(state,next_move) 
        path.append(next_state)

        if next_state == goal_state:
            done = True
        if step > maxSteps:
            done = True
        state = next_state
    return path


# In[301]:


def epsilonFindMoves(Q, state, epsilonRate, validMovesF):
    validMoveList = validMoves(state)
    if np.random.uniform()<epsilonRate:
        return validMoveList[np.random.choice(len(validMoveList))] 
    else:
        Qs = np.array([Q.get(stateMoveTuple(state, m), 0) for m in validMoveList])
        return validMoveList[np.argmin(Qs)]


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


# In[331]:


state = [[1],[],[2,3]]
print(validMoves(state))


# In[302]:


def makeMove(state, move):
    state2 = deepcopy(state)
    temp = state2[move[0]-1][0]
    state2[move[0]-1].pop(0)
    state2[move[1]-1].insert(0,temp)
    return state2


# In[12]:


def printState(state):
    temp = 0
    col1Len = 0
    col2Len = 0
    col3Len = 0
    big = len(max(state))
    while temp != len(max(state)):
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


# In[328]:


path = testQ(Q, 50, validMoves, makeMove)


# In[329]:


for x in path:
    printState(x)


# In[346]:


Q, stepsToGoal = trainQ(500, 0.5, 0.7, validMoves, makeMove)


# In[347]:


mean(stepsToGoal)


# In[341]:


mean(stepsToGoal)


# # Examples

# In[ ]:


state = [[1], [], [2,3]]
printState(state)


# In[ ]:


state = [[1,2,3],[],[]]
move = [1,2]
print(makeMove([[2], [3], [1]], [1, 2]))


# In[ ]:


state = [[1,2,3],[],[]]
print(validMoves(state))


# In[ ]:


state = [[1, 2, 3], [], []]
printState(state)


# In[ ]:


move =[1, 2]

stateMoveTuple(state, move)


# In[ ]:


newstate = makeMove(state, move)
newstate


# In[ ]:


printState(newstate)


# In[82]:


Q, stepsToGoal = trainQ(50, 0.5, 0.7, validMoves, makeMove)


# In[131]:


Q, steps = trainQ(1000, 0.5, 0.7, validMoves, makeMove)


# In[132]:


mean(steps)


# In[320]:


Q


# In[321]:


path


# In[10]:


for s in path:
    printState(s)
    print()


# ## Grading

# Download and extract `A4grader.py` from [A4grader.tar](http://www.cs.colostate.edu/~anderson/cs440/notebooks/A4grader.tar).

# In[343]:


get_ipython().magic('run -i A4grader.py')


# ## Extra Credit

# Modify your code to solve the Towers of Hanoi puzzle with 4 disks instead of 3.  Name your functions
# 
#     - printState_4disk
#     - validMoves_4disk
#     - makeMove_4disk
# 
# Find values for number of repetitions, learning rate, and epsilon decay factor for which trainQ learns a Q function that testQ can use to find the shortest solution path.  Include the output from the successful calls to trainQ and testQ.
