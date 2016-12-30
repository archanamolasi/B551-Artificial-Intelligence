# This program builds the spam classifier.
# Formulation of problem:
# We have used naive bayes and decision trees to classify the documents as spam/ non spam.
# Naive Bayes:We tested the accuracies for both continuous and binary features.
# In this program we have commented out the running of continuous features bayes classifier part.
# We have calulated the likelihood of every word in being a spam email and non spam email using the training dataset
# and the existence of a given word in spam documents( or frequency in case of continuous features). Based on the training
# also calculated the posteriors of being a spam or non spam email. We created a trained model and using that model file 
# and Naive bayes assumption of independence we calculated the probability of a given email to be spam or non spam.If spam
# probability is higher we classify it to be a spam email and vice-versa.

# Decision Trees: We tested the accuracies for both continuous and binary features.
# In this program we have commented out the running of continuous features decision tree part, because it was taking around 20 mins
# to generate decision tree for both binary and continuous features.
# The decision tree is built using binary features as folllows:
# Find the word with minimum entropy and label this as root.
# Recursively split the data on the minimum entropy word each time removing the minimum entropy word.
# If a label is found we assign the label to the node else recursively call the function.
# The left branch of the tree is for binary feature value 1, the righ tbranch is for binary feature value 0.

# Both binary and continuous features model show similar accuracies for naive bayes.
# The binary features model is better for decision trees.
# The accuracies for both the models is documented in Report.docx. 

#import modules
import os, math, sys, json, copy
import collections
from collections import OrderedDict
f=0
#stop_words=['a','an','the','to','from','has','have','is','you','me','with', 'by', 'id', 'for']
stop_words = ['a','b','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','esmtp','localhost','jan','feb','mar','apr','may','jun','jul','aug','sep','oct',
        't','u','v','w','y','z','id','smtp','mon','tue','wed','thu','fri','sat','sun','please','nov','dec','imap','countofspam','here','list',
        "of", "to", "and", "that", "is", "in", "i", "it", "not", "you",'the', 'a', 'be', 'this', 'are', 'for', 'as', 'have', "with", 
        "on",'was', 'he','she','right','even','must', 'they', 'by', 'what', 'all', 'do', 'there', 'who', "drive", 'such','very','its','see' ,'or' ,
        "or", 'would', 'about', 'his', 'so', 'no', 'your', 'some', 'me',  'more', 'just', 'like', 'say','make' ,'many','c',
        "my", 'which', 'm', 'when', 'their', 're', 'our', 'think', 'were','him', 'had', 'also','then','them',
        'out','may','time','am','get','up','than','new','mr','nntppostinghost','into', 'dos', 'use', 'this','that','them','his','her',
        'q','x','said', 'these' ,'go' ,'these', 'off', 'could', 'using', 'thanks','most','why','well','should',
        "lines", 'know', 'only', 'us', 'other', 'does', 'because', 'been', 'writes', 'how',
        "subject",'did','those','help','msg', "organization", "if", "at",  "can", "scsi", "but", "an", "any", "card", "one", "has", "will", "ide", 'from', 'we']

confusion_matrix = {"spam": [0, 0], "notspam": [0, 0]}        
conf_mat = {"spam": [0, 0], "notspam": [0, 0]} 
dec_tree = collections.OrderedDict()
dec_tree_cont = collections.OrderedDict()
#Node of the decision tree with its attributes
class Node:    
    def __init__ (self, word, count_spam_1, count_nspam_1, count_spam_0, count_nspam_0, label="" ):
        self.word = word
        self.count_spam_1 = count_spam_1
        self.count_nspam_1 = count_nspam_1
        self.count_spam_0 = count_spam_0
        self.count_nspam_0 = count_nspam_0
        self.left = None
        self.right = None
        self.label = label
#calculates accuracy from the confusion matrix
def calc_accuracy(conf_mat):
    acc_num=conf_mat["spam"][0] + conf_mat["notspam"][1]
    acc_den=conf_mat["spam"][1]+conf_mat["notspam"][0]+ acc_num
    print "Accuracy is:", (float(acc_num)/float(acc_den))*100, "%"
#calculates the count of spam emails and non spam emails in a given dataset directory and total document count    
def count(dataset_directory):
    countDoc=0
    label = []
    for filename in os.listdir("./"+dataset_directory+"/spam"):
        countDoc+=1
        label.append("S")
    for filename in os.listdir("./"+dataset_directory+"/notspam"):
        countDoc+=1
        label.append("NS")
    return countDoc, label
#creates word count dictionary per document
def read_docs_train(word_vec,directory):
    global f
    for filename in os.listdir(directory):
        with open(directory+filename,"r") as fr:
            txt=fr.read().split()
            for w in txt:
                w = w.lower()
                if w.isalpha():
                    word_vec[w][f]+=1
        f+=1 
#creates word count dictionary per document for binary and continuous features of naive bayes model      
def read_docs_train_bayes(directory,bin_cont_flag):
    word_vector=collections.defaultdict(lambda: 0)
    for filename in os.listdir(directory):
        seen=collections.defaultdict(lambda: 0)
        with open(directory+filename,"r") as fr:
            txt=fr.read().split()
            for w in txt:
                if w.lower() not in stop_words:
                    if  w.isalpha() and bin_cont_flag and not seen[w.lower()]:
                        seen[w.lower()]=1
                        word_vector[w.lower()]+=1
                    if  w.isalpha() and not bin_cont_flag:
                        word_vector[w.lower()]+=1
    return word_vector
#creates the model file after training the model on the trainig dataset and prints top spam/non spam words
def train_bayes(dataset_directory, model_file,bin_cont_flag):  
    #read spam_word_vec and nspam_word_vec from model file and create dicts based on binary and continuous models
    spam_word_vec=read_docs_train_bayes("./"+dataset_directory+"/spam/",1)
    nspam_word_vec=read_docs_train_bayes("./"+dataset_directory+"/notspam/",1)
    spam_word_vec_cont=read_docs_train_bayes("./"+dataset_directory+"/spam/",0)
    nspam_word_vec_cont=read_docs_train_bayes("./"+dataset_directory+"/notspam/",0)
    countDoc, label=count(dataset_directory)
    
    #write all the above probabilities to model file   
    model_dict=collections.defaultdict(dict)   
    for word in spam_word_vec:
        if word in nspam_word_vec:
            model_dict[word]["spam_prob"]=math.log(float(spam_word_vec[word])/float(label.count("S")))
        else:
            model_dict[word]["spam_prob"]=math.log(float(spam_word_vec[word])/float(label.count("S")))
            model_dict[word]["nspam_prob"]=math.log(1E-9)
    for word in nspam_word_vec:
        if word in spam_word_vec:
            model_dict[word]["nspam_prob"]=math.log(float(nspam_word_vec[word])/float(label.count("NS")))
        else:
            model_dict[word]["nspam_prob"]=math.log(float(nspam_word_vec[word])/float(label.count("NS")))
            model_dict[word]["spam_prob"]=math.log(1E-9)
    model_dict["countofspam"]["spam_prob"]=math.log(float(label.count("S"))/float(countDoc))
    model_dict["countofspam"]["nspam_prob"]=math.log(float(label.count("NS"))/float(countDoc))  
    model_dict_cont=collections.defaultdict(dict)   
    for word in spam_word_vec:
        if word in nspam_word_vec:
            model_dict_cont[word]["spam_prob"]=math.log(float(spam_word_vec_cont[word])/float(label.count("S")))
        else:
            model_dict_cont[word]["spam_prob"]=math.log(float(spam_word_vec_cont[word])/float(label.count("S")))
            model_dict_cont[word]["nspam_prob"]=math.log(1E-9)
    for word in nspam_word_vec:
        if word in spam_word_vec:
            model_dict_cont[word]["nspam_prob"]=math.log(float(nspam_word_vec_cont[word])/float(label.count("NS")))
        else:
            model_dict_cont[word]["nspam_prob"]=math.log(float(nspam_word_vec_cont[word])/float(label.count("NS")))
            model_dict_cont[word]["spam_prob"]=math.log(1E-9)
    model_dict_cont["countofspam"]["spam_prob"]=math.log(float(label.count("S"))/float(countDoc))
    model_dict_cont["countofspam"]["nspam_prob"]=math.log(float(label.count("NS"))/float(countDoc))
    
    if bin_cont_flag:
        with open(model_file, 'w') as f:
            json.dump(model_dict, f, indent = 4) 
    else:
        with open(model_file, 'w') as f:
            json.dump(model_dict_cont, f, indent = 4)
    
    topspam=OrderedDict(sorted(model_dict.iteritems(), key=lambda x: x[1]["spam_prob"],reverse=True))
    del topspam["countofspam"]
    print "Top 10 spam words are:\n",topspam.keys()[:10]
    topnspam=OrderedDict(sorted(model_dict.iteritems(), key=lambda x: x[1]["nspam_prob"],reverse=True))
    del topnspam["countofspam"]
    print "Top 10 non spam words are:\n",topnspam.keys()[:10]
    return
#test the test dataset and prints the confusion matrix and accuracy
def test_bayes(dataset_directory, model_file):  
    #read spam_prob and  from model file and create dicts
    model_dict=collections.defaultdict(dict)
    model_dict=json.load(open(model_file, 'r'))
    test_docs_bayes("./"+dataset_directory+"/spam/",model_dict,"spam")
    test_docs_bayes("./"+dataset_directory+"/notspam/",model_dict,"notspam")
    print "Format of confusion matrix: {notspam: [spam_count, notspam_count], spam: [spam_count, notspam_count]}"
    print( "Confusion matrix is ", conf_mat) 
    calc_accuracy(conf_mat)       
    return
#calculates the probability of a document being spam and non spam and assign proper labels.Creates confusion matrix.
def test_docs_bayes(directory,model_dict, dir_label):  
    prior_spam_prob=model_dict["countofspam"]["spam_prob"]
    prior_nspam_prob=model_dict["countofspam"]["nspam_prob"]
    for filename in os.listdir(directory):
        exp_prob_spam=0
        exp_prob_nspam=0
        with open(directory+filename,"r") as fr:
            txt=fr.read().split()
            for w in txt:
                if w.isalpha() and w.lower() in model_dict:
                    exp_prob_spam=exp_prob_spam+model_dict[w.lower()]["spam_prob"] 
                    exp_prob_nspam=exp_prob_nspam+model_dict[w.lower()]["nspam_prob"]
                elif w.isalpha() and w.lower() not in model_dict:
                    exp_prob_spam=exp_prob_spam+math.log(1E-9) 
                    exp_prob_nspam=exp_prob_nspam+math.log(1E-9)                   
            exp_prob_spam=exp_prob_spam+prior_spam_prob
            exp_prob_nspam=exp_prob_nspam+prior_nspam_prob 
            
            if exp_prob_nspam>exp_prob_spam:
                cls="NS"
            else:
                cls="S"  
                             
        if cls == "S":
            conf_mat[dir_label][0] += 1
        elif cls == "NS":
            conf_mat[dir_label][1] += 1 

# Return the label for each testing file.
def test_doc(dec_tree_key, dec_tree_test, word_vec):
    if dec_tree_key in ("S","NS", "") :
        return dec_tree_key
    if dec_tree_key in word_vec:
        return test_doc(dec_tree_test[dec_tree_key][0], dec_tree_test, word_vec)    
    else:
        return test_doc(dec_tree_test[dec_tree_key][1], dec_tree_test, word_vec)    
    return label

# Iterate over all testing documents and create confusion matrix.
def read_test_docs_test(directory, dec_tree_test, dir_label):        
    root = list(dec_tree_test)[0]
    for filename in os.listdir(directory):
        word_vec = collections.defaultdict(lambda: 0)
        with open(directory+filename,"r") as fr:
            txt=fr.read().split()
            for w in txt:
                w = w.lower()
                if w.isalpha():
                    word_vec[w] += 1
                     
        r = test_doc(root, dec_tree_test, word_vec)
        if r == "S":
            confusion_matrix[dir_label][0] += 1
        elif r == "NS":
            confusion_matrix[dir_label][1] += 1    

# Calculate the minimum entropy word fromthe word_vec.                     
def entropy(word_vec, label):
    min_entropy = sys.maxsize
    root = None
    for key in word_vec:
        count_spam_1 = 0.0
        count_spam_0 = 0.0
        count_nspam_1 = 0.0
        count_nspam_0 = 0.0
        for i in range(len(label)):
            if label[i] == "S":
                if word_vec[key][i] > 0:
                    count_spam_1 += 1
                else:
                    count_spam_0 += 1
            else:
                if word_vec[key][i] > 0:
                    count_nspam_1 += 1
                else:
                    count_nspam_0 += 1        
        total = count_spam_1 + count_spam_0 + count_nspam_1 +count_nspam_0
        count_1 = count_spam_1 +count_nspam_1
        count_0 = count_spam_0 +count_nspam_0    
        pspam_1 = 0        
        if count_1 > 0 and count_spam_1/count_1 >0:
            pspam_1 = -(count_spam_1/count_1)*math.log(count_spam_1/count_1,2)
        pspam_0 = 0
        if count_0 > 0 and count_spam_0/count_0>0:
            pspam_0=-(count_spam_0/count_0)*math.log(count_spam_0/count_0,2)            
        pnspam_1 = 0
        if count_1 > 0 and count_nspam_1/count_1>0:
            pnspam_1=-(count_nspam_1/count_1)*math.log(count_nspam_1/count_1,2)
        pnspam_0 = 0
        if count_0 > 0 and count_nspam_0/count_0>0:
            pnspam_0=-(count_nspam_0/count_0)*math.log(count_nspam_0/count_0,2)        
        ent = (count_1/total)*(pspam_1+pnspam_1) +(count_0/total)*(pnspam_0+pspam_0)    
        if ent <min_entropy:
            min_entropy = ent
            root=Node (key,count_spam_1,count_nspam_1,count_spam_0,count_nspam_0)
    return root

# Split the word_vec and label on the basis of word.
# The flag if set to True splits data for continuous features otherwise for binary features.
def split_data(word, word_vec, label, flag):
    indices_1 = []
    indices_0 = []
    word_vec_1 = collections.defaultdict(list)
    word_vec_0 = collections.defaultdict(list)
    label_1= []
    label_0 = []
    if flag == False: 
        for val in range(len(word_vec[word])):
            if word_vec[word][val] > 0 :
                indices_1.append(val)
                label_1.append(label[val])
            elif word_vec[word][val] == 0:
                indices_0.append(val)
                label_0.append(label[val])
    else:
        for val in range(len(word_vec[word])):
            threshold = int(sum(word_vec[word])/len(word_vec[word]))
            if word_vec[word][val] > threshold :
                indices_1.append(val)
                label_1.append(label[val])
            else:
                indices_0.append(val)
                label_0.append(label[val])     
 
    word_vec.pop(word)        
    for ran in xrange(len(indices_1)):
        for key in word_vec:
            word_vec_1[key].append(word_vec[key][indices_1[ran]])
    for ran in xrange(len(indices_0)):
        for key in word_vec:
            word_vec_0[key].append(word_vec[key][indices_0[ran]])
    return [word_vec_1,word_vec_0],[label_1,label_0]

# Builds the decision tree recursively.     
def entropy_recursive(root, word_vec, label, depth, flag)    :    
    #print ("word     ", root.word)
    word_vec_list, label_list = split_data (root.word, word_vec, label, flag)
    #print ("After split ", len (word_vec_list[0]), len(word_vec_list[1]))
    if len(word_vec_list[0]) == 0 or len(word_vec_list[1]) ==0:
        return None
    if root.count_spam_1 == 0:
        root.left=Node(None,-1,-1,-1,-1,"NS")
    elif root.count_nspam_1==0:
        root.left=Node(None,-1,-1,-1,-1,"S")
    else:
        r=entropy(word_vec_list[0],label_list[0])
        root.left = entropy_recursive(r,word_vec_list[0],label_list[0], depth+1, flag)         
    if root.count_spam_0 == 0:
        root.right=Node(None,-1,-1,-1,-1,"NS")
    elif root.count_nspam_0==0:
        root.right=Node(None,-1,-1,-1,-1,"S")
    else:
        r=entropy(word_vec_list[1],label_list[1])
        root.right= entropy_recursive(r,word_vec_list[1],label_list[1], depth+1, flag)
    return root

# Print the 4 levels of the decision tree.
def printLevelOrder(root):
    # Base Case
    if root is None:
        return
    # Create an empty queue for level order traversal
    queue = []
    # Enqueue Root and initialize height
    queue.append(root)
    count=15
    while((len(queue) > 0) and count>0):
        # Print front of queue and remove it from queue
        if queue[0].word:
            print queue[0].word
        else:
            print queue[0].label
        node = queue.pop(0)
        count-=1
        #Enqueue left child
        if node.left is not None:
            queue.append(node.left)
        # Enqueue right child
        if node.right is not None:
            queue.append(node.right)

# Create decision tree for binary features.
def printTree(root):    
    if root == None:     
        return    
    if root.word != None:        
        l = ""
        r = ""
        if root.left:
            l = root.left.label
            if root.left.word != None:
                l = root.left.word
        if root.right:
            r = root.right.label
            if root.right.word != None:
                r = root.right.word         
        dec_tree[root.word] = [l, r]           
    printTree(root.left)    
    printTree(root.right)
 
# Create decision tree for continuous features.   
def printTreeCont(root):    
    if root == None:  
        return                  
    if root.word != None:        
        l = ""
        r = ""
        if root.left:
            l = root.left.label
            if root.left.word != None:
                l = root.left.word
        if root.right:
            r = root.right.label
            if root.right.word != None:
                r = root.right.word         
        dec_tree_cont[root.word] = [l, r]         
    printTreeCont(root.left)    
    printTreeCont(root.right)    

 # Train the data and prepare model file.
def train_dt(word_vec, label, dataset_directory, model_file):    
    print "Training decision tree for binary features."
    read_docs_train(word_vec,"./"+dataset_directory+"/spam/")
    read_docs_train(word_vec,"./"+dataset_directory+"/notspam/")
    word_vec_copy = copy.deepcopy(word_vec)
    root = entropy(word_vec, label)
    root = entropy_recursive(root, word_vec, label, 0, False)
    print "Level order traversal of decision tree."
    printTree(root)
    printLevelOrder(root)
    # The following code can be used to generate decision tree for continuous features.
    #print "Training decision tree for continuous features."
    #root = entropy(word_vec_copy, label)
    #root = entropy_recursive(root, word_vec_copy, label, 0, True)
    #print "Level order traversal of decision tree."
    #printTreeCont(root)
    #printLevelOrder(root)
    #print dec_tree_cont
    #with open(model_file, 'w') as f:
    #    json.dump(dec_tree_cont, f, indent = 4)
    with open(model_file, 'w') as f:
        json.dump(dec_tree, f, indent = 4)

# Helper function to create list of zeroes, equal in length to the number of training documents.     
def def_lst():
    return [0]*docs

# Test data and print confusion matrix. 
def test_dt (dataset_directory, model_file):
    dec_tree_test = collections.OrderedDict()
    dec_tree_test = json.load( open(model_file, 'r'), object_pairs_hook=OrderedDict)   
    read_test_docs_test("./"+dataset_directory+"/spam/", dec_tree_test, "spam")
    read_test_docs_test("./"+dataset_directory+"/notspam/", dec_tree_test, "notspam")
    print "Format of confusion matrix: {notspam: [spam_count, notspam_count], spam: [spam_count, notspam_count]}"
    print( "Confusion matrix is ", confusion_matrix)
    calc_accuracy(confusion_matrix)

# Read from command line.
if __name__ == "__main__":    
    [mode, technique, dataset_directory, model_file] = sys.argv[1:5]
    docs, label=count(dataset_directory)
    word_vec=collections.defaultdict(def_lst)
    if mode == "train":
        if technique == "dt":                        
            train_dt(word_vec, label, dataset_directory, model_file)
        elif technique == "bayes"    :
            train_bayes(dataset_directory, model_file,1)
        else:
            print ("Unknown technique"    )
    elif mode == "test":
        if technique == "dt":            
            test_dt (dataset_directory, model_file)
        elif technique == "bayes"    :
            test_bayes(dataset_directory, model_file)
        else:
            print ("Unknown technique"    )
    else:
        print ("Unknown mode")