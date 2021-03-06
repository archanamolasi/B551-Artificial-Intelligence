#Archana Molasi, September 15

#1) This program solves the 15 puzzle. 
#Formulation of the search problem.
#Initial State: The initial state is the 4x4 board configuration given as input in the text file. 
#Goal State: The goal state is the canonical configuration of the board with tiles 1-15 arranged sequentially and empty tile in the
#lower right corner. The value of heuristic function will be 0 at goal state.
#State Space: The state space will contain all board configurations with an even permutation inversion generated by the successor 
#function. The states with odd permutation inversion are filtered out.
#Successor function: The successor function will slide tiles surrounding the empty tile one by one into the empty tile. Every state will
#have four successor #states because an empty tile can be filled from the four tiles surrounding it. This is true for edge and corner
#tiles as well. The branching factor is four.
#Edge weights: The edge weight of a state n is the total cost f(n), where f(n)=g(n)+h(n), g(n) is the cost of reaching the state n 
#from start state, h(n) is #the heuristic cost, the cost of reaching the goal state from the state n.

#Heuristic functions tried for this problem:
#Misplaced tiles: calculate_heuristic_misplaced_tiles(state) method commented in this code shows how to calculate the number of
#misplaced tiles. Let us denote this heuristic function as h1. This heuristic is admissible because h1(n)=1 for a misplaced tile,
#otherwise 0. Thus, it never overestimates the actual cost to the goal. However, this heuristic is not used in this code because 
#this doesn't give a heuristic cost closer to goal. 
#X-Y distance: calculate_xy(state) method commented in the shows how to calculate the X-Y value for a state.
#Source:https://heuristicswiki.wikispaces.com/X-Y, Let us denote this heuristic function as h2. This heuristic is admissible because 
#for every misplaced tile it adds the straight line distance to its row and column.
#Therefore, it will not be greater than the actual distance required to reach the goal. This heuristic is not used in this code because
#the performance of #the program was not improved by it.
#Manhattan distance: calculate_heuristic_manhattan(state) method  in this code calculates the total manhattan distance of the tiles in 
#a state. Let us denote this heuristic function as h3. This heuristic is admissible because for a tile it adds the minimum cost of
#reaching the correct position of the tile. Therefore, it will be less than the actual cost. This heuristic assumes that a tile is 
#isolated and free to move within the board, hence it will never overestimate the actual cost. This heuristic is used in this code.
#Linear conflict: calculate_linear_conflict(state) method  in this code calculates the linear conflict between tiles of a state.
#This heuristic is admissible because it adds two moves for each pair of conflicting tiles which are in their goal row, column
#but incorrectly placed such that they have to be moved by atleast two moves:
#Source: https://heuristicswiki.wikispaces.com/Linear+Conflict. Let us denote this heuristic function as h4.  

#Comparison of performance of heuristic functions:
#(h3+h4)>h1 and h2 >h1. h2 and (h3+h4) were almost comparable so only one of them is used. Please note that (h3+h4) is an
#admissible heuristic, because manhattan distance does not take into account the extra moves required for conflicting tiles.
#Source:
#http://stackoverflow.com/questions/35552661/can-linear-conflict-heuristic-cause-more-nodes-to-be-created-and-explored-than-m/35784864 

#2) The search begins with generating the successors of the initial state. Each new successor state is tested for goal state by
#comparing its heuristic value #to 0. The successor states are added to the fringe which is a dictionary with key =g(n)+h(n) i.e
#the total cost of the state; and value as the state #configuration stored as a list. The search is a modified DFS in which 
#during the pop operation the successor state with lowest key is picked to  get an optimal solution.
#In case, more than one state has the same key value, the leftmost state is popped first. 
#The search stops when we find a successor state with heuristic cost=0.

#3) This code gives optimal solution for all puzzles with initial permutation inversion<=40. For the tougher puzzles, speed is 
#given preference to optimality.
#This is done by multiplying the initial cost g(n) by a constant to favor a greedy approach. The constant which worked best was 0.5.
#eg: two nodes with initial cost 2 and 3 will become 1 each(0.5*2 and 3*0.5). Therefore node with cost 3 may be picked up before
#node with cost 2.
#http://stackoverflow.com/questions/23729144/a-finding-a-better-solution-for-15-square-puzzle-with-one-given-solution


# total tiles
total=16
#tiles per row/column
N=4

class Node:
    '''
    This class wraps up the properties of a state :
    state: the configuration of the board
    moves: the number of moves required to reach the board state above.
    cost: the initial cost g(n) required to reach state n.
    permutation_inversion: the permutation inversion of the state to identify solvable puzzles.
    heuristic_manhattan: the total heuristic cost h(n)of state n i.e manhattan distance+ linear conflict of the state. 
    '''
    def __init__(self, state,moves,cost,heuristic_manhattan,permutation_inversion):
        self.state = state
        self.moves = moves
        self.cost=cost
        self.permutation_inversion=permutation_inversions(self.state)  
        self.heuristic_manhattan=calculate_heuristic_manhattan(self.state) +calculate_linear_conflict(self.state)
             
#This method calculates the permutation inversion for a given state.
import math
def permutation_inversions(board):
    p=0
    zero=0   
    for i in xrange(0,total-1):
        element=board[i] 
        if element==0:
            zero=i//4+1
        for j in xrange(i+1,total):
            b=board[j]
            if b<element and b!=0:
                p+=1                      
    return p +zero 
    
coordinates=[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]    
#This method calculates linear conflict of the state.
def calculate_linear_conflict(board):
    linear_conflict=0 
    for i in xrange(0,total-1):
        element=board[i] 
        for j in xrange(i+1,total):
            b=board[j]
            if b<element and b!=0:               
                should_be=coordinates[element-1]
                also_should_be=coordinates[b-1]                
                element_current=[i//4,i%4]
                board_j_current=[j//4,j%4]        
                if(should_be==board_j_current and also_should_be==element_current):
                        linear_conflict+=2 
    return linear_conflict              
    
#This method calculates the manhattan distance of the state.    
def calculate_heuristic_manhattan(state):
    h=0
    for i in xrange(0,total):
            if state[i]!=0:
                x=math.fabs(coordinates[i][0]-coordinates[state[i]-1][0])
                y=math.fabs(coordinates[i][1]-coordinates[state[i]-1][1])
                h+=x+y
    return int(h) 
#This method calculates number of misplaced tiles of the state.    
#def calculate_heuristic_misplaced_tiles(state):
#    h=0
#    for i in range(0,total-1):
#        if state[i]!=i+1 and state[i]!=0:
#            h+=1
#    if state[total-1]!=0:
#        h+=1 
#    return h    

#This method calculates x-y distance of the state.
#def calculate_xy(state):
#    rows=0
#    columns=0
#    for i in xrange(0,total):
#        current=i//4
#        should_be=coordinates[state[i]-1][0]
#        rows+=math.fabs(should_be-current)
#    for i in xrange(0,total):
#        current=i%4
#        should_be=coordinates[state[i]-1][1]
#        columns+=math.fabs(should_be-current)    
#    return int(rows+columns)  

#This method reads the input file and returns a list (initial board).
def initial_state(fileName):   
    board=[]
    try:  
        file=open(fileName,mode='r')
        for line in file.readlines():
            board.extend(int(i) for i in line.split()) 
        print "Initial board",board
        file.close()
    except IOError:
        print "Cannot read from file:"
        return
    return board
#The successor function, returns a dictionary of states with key as the move and value as the board configuration reached from that move.
import copy
def successor(board):
    states={}
    zero=board.index(0 )
    #up
    states['U']=(swap_items(board, zero, (zero+N)%total))   
    #down
    states['D']=(swap_items(board, zero, zero-N) if zero-N>=0 else swap_items(board, zero, zero+total-N-1) )
    #left
    states['L']=(swap_items(board, zero, zero-N-1) if zero%N==N-1 else swap_items(board, zero, zero+1))
    #right
    states['R']=(swap_items(board, zero, zero+N-1) if zero%N==0 else swap_items(board, zero, zero-1))
    return states
    
#This method is a helper function for successor function. It is used for creating the new successor state.
def swap_items(board,a,b):
    copy_board=copy.copy(board)
    copy_board[a],copy_board[b]=copy_board[b],copy_board[a]
    return copy_board

#This is the search function, which maintains the fringe, a dictionary{total cost: state}, list of visited states.
#A modified DFS is performed such that the state with lowest key is popped first from the fringe.
def solve_puzzle(board):  
    import collections
    fringe=collections.defaultdict(list)
    visited=[]    
    #A constant to multiply with cost g(n) for tough boards.
    greedy=1
    obj=Node(board,[],0,0,0)
    min_heuristic=obj.heuristic_manhattan
    fringe[min_heuristic].append(obj)
    initial_permutation=obj.permutation_inversion
    if initial_permutation%2==1:
       print("Enter a valid input, this input is not solvable.")
    #Boards with initial permutation inversion>40 are considered tough boards, hence greedy constant is set to 0.5 to modify the cost g(n).
    if initial_permutation>40:
       greedy=0.5    
    while(len(fringe) ): 
        minimum=min(key for key in fringe.keys())  
        #If a fringe key has more than one state, return the leftmost state, hence the 0 in pop().
        get_element=fringe[minimum].pop(0)
        if not fringe[minimum]:
            del fringe[minimum]               
        element=get_element.state
        moves=get_element.moves  
        cost=get_element.cost       
        p=get_element.permutation_inversion    
        if element not in visited and p%2==0 :
            temp=successor(element)           
            for k,v in temp.items(): 
                        new_moves=copy.copy(moves)            
                        new_moves.append(k)          
                        c=cost+1                        
                        obj=Node(v,new_moves,c,0,0)  
                        hm=obj.heuristic_manhattan
                        #If goal state.                      
                        if hm==0 :
                           print "Goal reached",v
                           print " ".join(move for move in new_moves )
                           #print(len(new_moves))
                           return 0
                        p=obj.permutation_inversion                        
                        k=c*greedy+hm
                        if p%2==0 :
                            fringe[k].append(obj)                                                
        visited.append(element)
    return "Fail" 

import sys     
x=sys.argv[1:]
if x: 
    fileN=x[0] 
    board=initial_state(fileN)    
    solve_puzzle(board) 
else:
    print "Error, enter input file name."    