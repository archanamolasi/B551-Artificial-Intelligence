#Pass the input through command line delimited by a space.
#Eg: python Yahtzee.py 1 2 3
#Expected value of sum of 3 dice= 3.5+3.5+3.5=10.5
#If the sum of the three dice is less than 11, we re-roll according to different cases.

#Function to calculate re-rolls.
def yahtzee(inputList):    
    sumOfNumbers=sum(number for number in inputList)
    from collections import Counter
    countList=Counter(inputList)
    inputList.sort()
    if 3 in countList.values():
        print "Don't re-roll. You have already won"
    elif 2 in countList.values():
        if sumOfNumbers<11  and (inputList[1]+inputList[2])<7 :
            if countList[1]==2 or countList[2]==2:
                print "Re-roll three dice."
            else:
                print "Re-roll the dice with least two numbers",inputList[0]," ",inputList[1]
        elif sumOfNumbers<=11  and (inputList[1]+inputList[2])>=7 :
            print "Re-roll die with number" ,inputList[0]
        else:
            print "Don't Re-roll any dice."   
    else:
        if sumOfNumbers<=11  and (inputList[1]+inputList[2])<7 :
            print "Re-roll the three dice"
        elif sumOfNumbers<=11  and (inputList[1]+inputList[2])>=7 :
            print "Re-roll the least two dice.",inputList[0],inputList[1]
        else:
            print "Don't Re-roll any dice."     

import sys
try:
    inputList=map(int,raw_input().strip().split())
    if len(inputList) != 3:
         print "Input error"  
    else:
        yahtzee(inputList)    
except IOError:
    print "Input error"
