###################################
# CS B551 Fall 2016, Assignment #3
#
# Archana Molasi (molasia)
# Prerna Preeti (ppreeti)
# Shruti Rachh (srachh)
# (Based on skeleton code by D. Crandall)
#
#
####
# Report:
# 1. a. 
# W(i) =[W1,W2,..Wn] are the set of words given in the sentence.
# S(i) =["noun","verb",.."adj"] are the part of speech labels.
# The initial probablities are computed as P(S(i)):
#     P(S(i)) = count of state Si occurring at the beginning of the sentence/count of total number of sentences.

# The transition probabilities are computed as P(S(i+1)|S(i)):
#     P(S(i+1)|S(i)) = count of occurrences where S(i+1) comes after S(i)/count of occurences of S(i)

# The emission probabilities are computed as P(W(i)|S(i)):
#     P(W(i)|S(i)) = count of occurences where W(i) is labeled as S(i)/count of occurences of S(i)

# 1. b.
# Simplified: This model does not incorporate much evidence and each observed variable depends only on the current state.
# There is no dependency between the hidden states. To calculate the best tag sequence, at each time t we calculate the emission probabilities for the word with all the possible tags. We pick the maximum emission probability and the tag corresponding to it is stored as the tag for that word.

# 1. c.
# We have implemented Viterbi algorithm using dynamic programming.
# We compute the probabilities for all labels for first word in the sentence as:
#     v[0]={S(i):[P(S(i)*P(W1|S(i)),S(i)]}
# Next, we compute the probabilities for all remaining words as:
#     v[i]={S(i):[max j(v[i-1]*P(S(j+1)|S(j)))*P(W(i)|S(i)),S(j)]}
# where S(j) is the label for previous word which gives the maximum probability for current word.
# After computing all probabilities, we backtrack using the stored S(j) label and we get the most probable parts of speech tags for the given sentence. 

# 1. d.
# Complex: This model incorporates more information about a state because each state depends upon two previous states.
# We have implemented Forward algorithm to find the best sequence for the complex bayes net.
# For each state i at time t we calculate forward probability a(t) at state i = sum(a(t-1) * P(Si|Sj) * P(Ot|Si)) for all j in the domain of hidden states. This algorithm uses dynamic approach as we store the forward probabilities for each observed variable and tag combination.
# After the calculation for all observed variables is done, we backtrack picking the maximum value of a for each observed variable.
# The complex model will not give better results if the test data has lot of unseen words.

# Assumptions:
# To handle unseen emission and transition probabilities for unknown words, we are performing smoothing by taking 
# the unknown probabilities as 1e-10. We tried additive smoothing techniques like laplace add one smoothing and lambda smoothing, but the
# results were not impressive.

# Accuracy of the results:
# ==> So far scored 2000 sentences with 29442 words.
#                    Words correct:     Sentences correct: 
#    0. Ground truth:      100.00%              100.00%
#      1. Simplified:       93.92%               47.45%
#             2. HMM:       95.07%               54.25%
#         3. Complex:       91.91%               38.65%
# ----
# We have pushed Results.txt file which contains the detailed results of the algorithms on the testing file bc.test.

####
from __future__ import division
import random
import math
import time
import collections
from collections import Counter
import sys

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    #la=1
    initProb = collections.defaultdict(lambda:0)
    transProb = collections.defaultdict(lambda:0)
    emission_prob = collections.defaultdict(lambda:0)
    wordCnt=0
    totCnt={}
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        postProb=math.log (self.totCnt[label[0]]/sum(self.totCnt.values()),2)
        for l in range(len(label)-1):
            e=math.log (1e-10,2)
            t=math.log(1e-10,2)
            if (sentence[l],label[l]) in self.emission_prob:
                e=math.log(self.emission_prob[(sentence[l],label[l])],2)
            if (label[l+1],label[l]) in self.transProb:    
                t=math.log(self.transProb[(label[l+1],label[l])],2)
            postProb+=e+t
        emission = math.log(1e-10,2)
        if (sentence[-1],label[-1]) in self.emission_prob:
            emission = math.log(self.emission_prob[(sentence[-1],label[-1])]    ,2)
        return postProb+emission

    def compute(self,data):
        self.totCnt = Counter(cnt for d in data for cnt in d[1])

    def getTransProb(self, data):
        for i in range(len(data)):
            for j in range (len(data[i][1]) - 1):
               self.transProb [(data[i][1][j+1], data[i][1][j])]  += 1
        self.transProb = {key: float(cnt)/self.totCnt[key[1]] for key,cnt in self.transProb.iteritems()}

    def getInitialProb(self,data):
        dataLen=len(data)
        for d in data:
            self.initProb[d[1][0]]+=1
        total = sum(self.initProb.itervalues(), 0.0) 
        self.initProb = {key: float(cnt)/total for key,cnt in self.initProb.iteritems()}
    
    def getEmProb(self,data):
        vocab=set()
        for i in range(len(data)):
            for j in range (len(data[i][0]) ):
               vocab.add(data[i][0][j])
               self.emission_prob [(data[i][0][j], data[i][1][j])]  += 1
        self.wordCnt=len(vocab)
        self.emission_prob = {key: float(cnt)/self.totCnt[key[1]] for key,cnt in self.emission_prob.iteritems()}

    # Do the training!
    #
    def train(self, data):
        self.getInitialProb(data)
        self.compute(data)
        self.getTransProb(data)
        self.getEmProb(data)
    
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        total=sum(self.totCnt.values())
        tags=[]
        prob=[]
        for w in sentence:
            maxProb=-sys.maxsize
            state=""
            for s in self.initProb:
                p=1e-10
                if (w,s) in self.emission_prob:
                    p=self.emission_prob[(w,s)]
                p*=(float(self.totCnt[s])/total)
                if p>maxProb:
                    maxProb=p
                    state=s
            tags.append(state)
            prob.append(maxProb)
        return [ [tags], [prob] ]

    def hmm(self, sentence):
        v=[]
        viterbi={}
        for tag in self.initProb:
            viterbi[tag]=[1e-10,tag]
            if (sentence[0],tag) in self.emission_prob:
                viterbi[tag]=[self.initProb[tag]*self.emission_prob[(sentence[0],tag)],tag]
           
        v.append(viterbi)

        for i in range(1,len(sentence)):
            viterbi={}
            for j in self.initProb:
                maxVal=-sys.maxsize
                maxTag=j
                for k in self.initProb:
                    p=1e-10
                    if (j,k) in self.transProb:
                        p=v[i-1][k][0]*self.transProb[(j,k)]
                    if maxVal<p and p!=1e-10:
                        maxVal=p
                        maxTag=k
                if maxVal==-sys.maxsize:
                    maxVal=1e-10
                viterbi[j]=[(maxVal*1e-10),maxTag]
                if (sentence[i],j) in self.emission_prob:
                    viterbi[j]=[maxVal*self.emission_prob[(sentence[i],j)],maxTag]
            v.append(viterbi)

        result = collections.deque ()
        maxVal = -sys.maxsize
        maxTag = ""
        lastTag =""
        for i in v[-1]:
            p = v[-1][i][0]
            if maxVal < p:
                maxVal = p
                maxTag = v[-1][i][1]
                lastTag = i
        if len(sentence)>1:
            result.appendleft(lastTag)  
        result.appendleft(maxTag)      
                    
        for i in range (len(v)-2, 0, -1) :
            result.appendleft(v[i][maxTag][1])  
            maxTag = v[i][maxTag][1]
        return [ [result], [ ]]

    def complex(self, sentence):
        forward_prob=[]
        fp={}
        for tag in self.initProb:
            fp[tag]=[1e-10,tag]
            if (sentence[0],tag) in self.emission_prob:
                fp[tag]=[self.initProb[tag]*self.emission_prob[(sentence[0],tag)],tag]
        forward_prob.append(fp)

        for i in range(1,len(sentence)):
            fp={}
            for j in self.initProb:
                fp[j]=[1e-10,j]
                p=1e-10
                for k in self.initProb:
                    if (j,k) in self.transProb:
                        p=p + forward_prob[i-1][k][0]*self.transProb[(j,k)]
                if (sentence[i],j) in self.emission_prob:       
                    fp[j]= [p*self.emission_prob[(sentence[i],j)],j]
            forward_prob.append(fp)    
        result = collections.deque ()
        prob = collections.deque ()
        maxVal = -sys.maxsize
        maxTag = ""
        for i in range (len(forward_prob)-1, -1, -1) :
            maxVal = -sys.maxsize
            for j in forward_prob[i]:
                p = forward_prob[i][j][0]
                if p > maxVal and p !=1e-10 :
                    maxVal = p
                    maxTag = j
            if maxVal == -sys.maxsize: 
                maxVal= 1e-10
                maxTag=j    
            result.appendleft(maxTag)        
            prob.appendleft(maxVal)
        return [ [ result], [prob] ]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

