#!/usr/bin/python
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016

#1-HMM Formulation
#We formulated this problem as HMM where all emission probabilities are given by the edge_strength mask for a particular point in the image.
#All transition probabilities are inversely proportional to the distance between previous point and current point of image.
#Transition probabilities are directly proportional to the similarity of contrast of previous point and current point.

#How Program works
#For HMM we used Gibbs sampling and generate N samples of the ridge-line and use the coordinates which occur most frequently and create the final ridge-line.
#For Gibbs sampling we start with generating a random column coordinate and find the max edge-weight row for that column. Based on this particular point we 
#calculate emission probabilities and transition probability and using those we find out the next most probable coordinates for next column and so on.
#If starting point is given by human we use that instead of random starting coordinates.

#Assumptions and challenges
#For part 3 we decrease the probability of coordinates  which lie much further than the point of reference.This helps to increase probability of points 
#which are closer and similar to reference point.

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import copy
import random
import math
from collections import *
# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2
# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image
#This function will implement section 1 of the assignment question 2.
#It calculates the maximum edge weight for each column of the image matrix and creates the ridge based on those y co-ordinates.
def bayesnet(edge_strength):
    max_contrast_row=[]
    es=copy.deepcopy(edge_strength)
    a=array(es)
    max_contrast_row=list(a.argmax(axis=0))
    return max_contrast_row
#This function calculates the sum of edge weights for all rows for a given column. This will be used later to calculate probability distribution.
def sum_gradient(edge_strength,col):
    es=copy.deepcopy(edge_strength)
    a=array(es)
    sumlist=sum(es,axis=0)
    return sumlist[col]
    
#this function finds the best next set of coordinates ahead of the previous point after calculating the probabilities.
def transition_ahead(edge_strength,col,sample,gt_row,gt_col):
    em_prob=[]
    row=sample[col-1]
    sumg=sum_gradient(edge_strength,col)
    if gt_row==0 and gt_col==0:
        for row1 in range(0,edge_strength.shape[0]):
            if(row1!=row):
                em_prob.append(edge_strength[row1][col]/sumg*1/abs(row1-row))
            else:
                em_prob.append(edge_strength[row1][col]/sumg)
    else:
        for row1 in range(0,edge_strength.shape[0]):
            if row1!=row and abs(row1-row)<3:
                em_prob.append(edge_strength[row1][col]/sumg*1/abs(row1-row))
            elif row1==row:
                em_prob.append(edge_strength[row1][col]/sumg)
            elif row1!=row and abs(row1-row)>2:
                em_prob.append(edge_strength[row1][col]/sumg*1/abs(row1-row)*(1/100))
        
    em_prob=array(em_prob)
    max=em_prob.argmax(axis=0)
    return max,col

#this function finds the best next set of coordinates before of the previous point after calculating the probabilities.
def transition_before(edge_strength,col,sample,gt_row,gt_col):
    em_prob=[]
    row=sample[col+1]
    sumg=sum_gradient(edge_strength,col)
    if gt_row==0 and gt_col==0:
        for row1 in range(0,edge_strength.shape[0]):
            if row1!=row:
                em_prob.append(edge_strength[row1][col]/sumg*1/abs(row1-row))
            else:
                em_prob.append(edge_strength[row1][col]/sumg) 
    else:
        for row1 in range(0,edge_strength.shape[0]):
            if row1!=row and abs(row1-row)<3:
                em_prob.append(edge_strength[row1][col]/sumg*1/abs(row1-row))
            elif row1==row:
                em_prob.append(edge_strength[row1][col]/sumg)
            elif row1!=row and abs(row1-row)>2:
                em_prob.append(edge_strength[row1][col]/sumg*1/abs(row1-row)*(1/100))
                  
    em_prob=array(em_prob)
    max=em_prob.argmax(axis=0)
    return max,col


#This function creates one sample of the ridge by generating a random column if coordinates of ridge point are not given else it uses
#the given coordinates to generate a sample.    
def sample(edge_strength,gt_row,gt_col):
    sample={}
    ridge=[]
    if gt_row==0 and gt_col==0:
        rand_col=random.randint(0,edge_strength.shape[1])
        es=copy.deepcopy(edge_strength)
        a=array(es)
        max_contrast_col=list(a.argmax(axis=0))
        sample[rand_col]=max_contrast_col[rand_col]
    else:
        rand_col=gt_col
        sample[rand_col]=gt_row
    for col in range(rand_col+1,edge_strength.shape[1]):
        next_row,next_col=transition_ahead(edge_strength,col,sample,gt_row,gt_col)
        sample[next_col]=next_row
    for col in range(rand_col-1,-1,-1):
        prev_row,prev_col=transition_before(edge_strength,col,sample,gt_row,gt_col)
        sample[prev_col]=prev_row
    for key in sorted(sample):
        ridge.append(sample[key])
    return ridge


#Calculates max frequency element in a list
def max_freq(list):
    count=Counter(list)
    return count.most_common()[0][0]


#This function will generate N samples of ridge and will finally create a final ridge list based on maximum frequency element in each column.
def gibbs_sampling(edge_strength,gt_row,gt_col):
    ridge=[]
    final_ridge=[]
    final=defaultdict(list)
    for i in range(20):
        ridge=sample(edge_strength,gt_row,gt_col)
        for col in range(0,edge_strength.shape[1]):
            final[col].append(ridge[col])
    for key in final:
        final_ridge.append(max_freq(final[key]))
    return final_ridge

# main program
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]
# load in image 
input_image = Image.open(input_filename)
# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)
#ridge=sample(edge_strength)
ridge1=bayesnet(edge_strength)
ridge2=gibbs_sampling(edge_strength,0,0)
ridge3 =gibbs_sampling(edge_strength,int(gt_row),int(gt_col))
# output answer
#bayes net 1(a)
imsave(output_filename, draw_edge(input_image,ridge1, (255, 0, 0), 5))
#HMM 1(b)
imsave(output_filename, draw_edge(input_image,ridge2, (0, 0, 255), 5))
#HMM with human feedback 1(c)
imsave(output_filename, draw_edge(input_image,ridge3, (0, 255, 0), 5))
