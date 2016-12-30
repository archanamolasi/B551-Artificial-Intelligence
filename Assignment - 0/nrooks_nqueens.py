# nrooks_nqueens.py : Solve the N-Rooks and N-Queens problem!
# Adapted from nrooks.py by D. Crandall, August 2016
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.
# The N-queens problem is: Given an empty NxN chessboard, place N queens on the board so that no queens
# can take any other, i.e. such that no two queens share the same row or column or diagonal.
# This is N, the size of the board.
# Maximum N=181, for only rooks to be generated.
# Maximum N=13 for both rooks and queens to be generated.
N=14

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]
    
# Original successors function.
def successors(board):
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]
    
# Modified successors to successors2.
def successors2(board):
    return_states=[]
    c_pieces=count_pieces(board)
    #If a board already has N rooks return.
    if c_pieces==N:
      return return_states
    for r in range(0, N) :
     for c in range(0,N):
      state_generated=add_piece(board, r, c)
      #If condition to ensure that a state which is exactly same as the parent state(here board) is not added.     
      if(state_generated !=board):
       return_states.append(state_generated)
    return return_states   
    
# Modified successors2 to successors3.
def successors3(board):
    row_checker=[]
    col_checker=[]
    return_state=[] 
    #Store positions where there is already a rook.    
    for r in xrange(0, N) :
     for c in xrange(0,N):
      if board[r][c]==1:
       row_checker.append(r)
       col_checker.append(c)   
    c_pieces=count_pieces(board)+1   
    if c_pieces <= N:
     #Generate states with no two rooks on same row or column.    
     for r in xrange(0, N) :
      for c in xrange(0,N):
       if r not in row_checker and c not in col_checker:
        state_generated=add_piece(board, r, c)
        if(state_generated !=board):
         return_state.append(state_generated)
    return return_state

# Get list of successors of given board state for n-queens,
import copy
def nqueen_successors3(board):
    row_checker=[]
    col_checker=[]
    diagonal_list=[]
    return_state=[]
    #Store positions where there is already a queen. 
    for row in xrange(0,N):
     for col in xrange(0, N) :
      if board[row][col]==1:
       row_checker.append(row)
       col_checker.append(col)
       diagonal_list.extend(diagonal_generator(row,col))
    c_pieces=count_pieces(board)+1 
    if c_pieces <= N:
     #Generate states with no two queens on same row or column or diagonal.   
     for r in xrange(0,N):
      for c in xrange(0, N) :
       if r not in row_checker and c not in col_checker and [r,c] not in diagonal_list:
        flag=True
        state_generated=add_piece(board, r, c)     
        if(state_generated !=board): 
         if c_pieces>=N//2 :
          r1=copy.copy(row_checker) #Syntax of copy taken from https://docs.python.org/2/library/copy.html
          r1.append(r)
          c1=copy.copy(col_checker)
          c1.append(c)
          d1=copy.copy(diagonal_list)
          d1.extend(diagonal_generator(r,c))
          flag= check_last_queens(r1,c1,d1)                      
        if (flag):           
            return_state.append(state_generated)             
    return return_state   

#Return diagonal list of a position [row,col]    
def diagonal_generator(row,col):
       diagonal_list=[]
       r=row
       c=col
       while(r>0 and c>0):
        r-=1
        c-=1
        diagonal_list.append([r,c])
       r=row
       c=col
       while(r<N-1 and c<N-1):
        r+=1
        c+=1
        diagonal_list.append([r,c])
       r=row
       c=col
       while(r<N-1 and c>0):
        r+=1
        c-=1
        diagonal_list.append([r,c])
       r=row
       c=col
       while(r>0 and c<N-1):
        r-=1
        c+=1
        diagonal_list.append([r,c]) 
       return diagonal_list     

#Forward-checking as soon as number of Queens in the state reaches >= N/2.
#Remove states which make it impossible for any other Queen to be placed in future.
#Based on pg 84 of the book.
#Read the concept of forward checking from https://en.wikipedia.org/wiki/Look-ahead_(backtracking)
def check_last_queens(row_checker,col_checker,diagonal_list):  
    flag=True
    for r in xrange(0,N):
      if r not in row_checker:
        flag=False
        for col in xrange(0,N):
         if col not in col_checker:
          if [r,col] not in diagonal_list:
           flag= True            
        if not flag:#Return as soon as a row is found where no Queen can be placed.
         return False     
    return flag
    
# Check if board is a goal state for both n-rooks and n-queens.
# No explicit checking for n-queens diagonals as we are not generating states with conflicting diagonals.
def is_goal(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in xrange(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in xrange(0, N) ] )
        
# Solve n-rooks!
from collections import deque
def solve(initial_board):
    dict_list={}
    #dict_list[0] to store unvisited states.
    #dict_list[1] to store visited states.
    dict_list[0]=deque([initial_board])
    dict_list[1]=[]
    while (dict_list[0]):
        element=dict_list[0].pop() 
        if is_goal(element):
             return(element)
        if element not in dict_list[1] :
         for s in successors3(element ):
          dict_list[0].append(s)
         dict_list[1].append(element)
    return False

# Solve n-queens!
def nqueen_solve(initial_board):
    dict_list_nqueen={}
    #dict_list_nqueen[0] to store unvisited states.
    #dict_list_nqueen[1] to store visited states.
    dict_list_nqueen[0]=deque([initial_board])
    dict_list_nqueen[1]=[]
    while (dict_list_nqueen[0]):   
        element=dict_list_nqueen[0].pop() 
        if is_goal(element):
            return(element)        
        if element not in dict_list_nqueen[1]:       
         for s in nqueen_successors3(element ):            
            dict_list_nqueen[0].append(s) 
         dict_list_nqueen[1].append(element) 
    return False


initial_board = [[0]*N]*N
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solve(initial_board)
print ("N-rooks")
print (printable_board(solution) if solution else "Sorry, no solution found. :(")

solution_nqueen = nqueen_solve(initial_board)
print ("N-queens")
print( printable_board(solution_nqueen) if solution_nqueen else "Sorry, no solution found. :(")

