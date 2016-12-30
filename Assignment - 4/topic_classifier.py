
#1 description of how you formulated the problem, including precisely defining the abstractions
# This program builds a topic classifier.
# We have used bayesian classifier technique to create a topic classifier.We build a training model to predict the topic of an unknown document by
#using the labeled documents.
#We then use this model to predict the labels for unknown documents.

#(2) a brief description of how your program works
# we are counting the frequency of each word occurring in a document which belongs to a particular topic to calculate the likelihood.
#We are using only the given fraction of documents(command line) from each topic to train the model and calculate our posteriors. Using 
#the Naive bayes assumption. We label The rest of the documents from each topic using our probabilities that we calculated while training.
#After this we use 100% of the training data to train our model and use this trained model to predict the topic label for our training data.
#we resample the training model for some iterations till the model converges.
#For testing we use this converged model to predict topics for given unlabeled documents and calcualte the accuracy of this model based on 
#correctly classified topics.

#(3) Assumptions made
#If the word is not found in the training data , we assume that the probability of that word to exist in topic is natural logarithm of (1E-9)

#(4)Answers to questions asked
#All the accuracies calculated using different fractions have been tabulated in other report file Report.docx.

#import modules
import collections, random, math, sys, os, json
stop_words = ['a','b','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s',
        't','u','v','w','y','z',
        "of", "to", "and", "that", "is", "in", "i", "it", "not", "you",'the', 'a', 'be', 'this', 'are', 'for', 'as', 'have', "with", 
        "on",'was', 'he','she','right','even','must', 'they', 'by', 'what', 'all', 'do', 'there', 'who', "drive", 'such','very','its','see' ,'or' ,
        "or", 'would', 'about', 'his', 'so', 'no', 'your', 'some', 'me',  'more', 'just', 'like', 'say','make' ,'many','c',
        "my", 'which', 'm', 'when', 'their', 're', 'our', 'think', 'were','him', 'had', 'also','then','them',
        'out','may','time','am','get','up','than','new','mr','nntppostinghost','into', 'dos', 'use', 'this','that','them','his','her',
        'q','x','said', 'these' ,'go' ,'these', 'off', 'could', 'using', 'thanks','most','why','well','should',
        "lines", 'know', 'only', 'us', 'other', 'does', 'because', 'been', 'writes', 'how',
        "subject",'did','those','help','msg', "organization", "if", "at",  "can", "scsi", "but", "an", "any", "card", "one", "has", "will", "ide", 'from', 'we']
topics_static = ['christian', 'motorcycles', 'autos', 'religion', 'windows', 'medical', 'space', 'crypto', 'xwindows', 'atheism', 'pc', 'mac', 'baseball', 'hockey', 'mideast', 'graphics', 'politics', 'electronics', 'forsale','guns']

# Creates a topic label for each file after calculating the probabilities from the word vector of each file and model vector 
# retrieved from the model file.
def test_file(word_vec, model_vector):    
    max_prob = -sys.maxsize
    max_label = ""        
    for topic in topics_static:
        prob = 0
        for w in word_vec:
            if w in model_vector[topic]:
                prob += model_vector[topic][w]
            else:
                prob += math.log(1E-9)    
        posterior = model_vector["CountPosterior"][topic]
        if posterior != 0:        
            prob +=math.log(float(posterior))    
        if prob > max_prob:
            max_prob = prob
            max_label = topic        
    return max_label        
# test all the files in the given directory using the model vector retrieved from the model file.Creates a 
# word vector for each file and passes to test_file method to assign topic label. Also creates a confusion matrix 
# based on true topic label and assigned label.Prints accuracy.
def test(model_vector, directory):
    true_label = collections.defaultdict(lambda:0)
    result = collections.defaultdict(lambda:0)    
    confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for topic in topics_static:
        for topic1 in topics_static:
            confusion_matrix[topic][topic1] = 0
    total = 0.0
    correct = 0.0
    for folder in os.listdir(directory):
        if os.path.isdir(directory+"/"+folder):
            for file in os.listdir(directory+"/"+folder):    
                word_vec = []        
                with open(directory+"/"+folder+"/"+file,"r") as fr:
                    word=fr.read().split()
                    for w in word:
                        w = w.lower().translate(None, "!@#$%^&*_-+=|\\;\/:.,<>?[]{}()")
                        if w not in stop_words and w.isalpha():
                            word_vec.append(w)
                label = test_file(word_vec, model_vector)            
                true_label[folder] += 1
                result[label] += 1
                confusion_matrix[folder][label] += 1
                if folder == label:
                    correct += 1
                total += 1    
    print "The format of confusion matrix is {topic1: {topic1:count_of_files_classified_in_topic1}, {topic2:count_of_files_classified_in_topic2}...{topic20:count_of_files_classified_in_topic20}} "
    print "The confusion matrix is: "
    for topic in confusion_matrix:
        print '{:12s}'.format (topic),"{",
        for labels in confusion_matrix[topic]:
            print labels,":", confusion_matrix[topic][labels] ,
        print "}"     
        print "\n"
    print "Total files ", total
    print "Correctly classified files ", correct
    print "Accuracy is", (float(correct/total) * 100)
    
# creates a dictionary for topics and their file counts        
def create_count(directory):
    count_topics = collections.defaultdict(lambda: 0)
    for folder in os.listdir(directory):
        if os.path.isdir(directory+"/"+folder):
            for file in os.listdir(directory+"/"+folder):            
                count_topics [folder] += 1
    return count_topics

bul=0.00001


#Creates a nested dictionary and stores word and thier counts per topic per file.
#{topic1:{"file1":{"word1":32,"word2":2,.....}}}
#This file vector is later used to count probabilities of topic given words in a file.
def create_file_vec(directory):
    word_vec =  collections.defaultdict(lambda: collections.defaultdict(lambda:0 ))
    file_vec =  collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda:0 ) ))    
    for folder in os.listdir(directory):
        key = 0
        if os.path.isdir(directory+"/"+folder):
            for file in os.listdir(directory+"/"+folder):            
                with open(directory+"/"+folder+"/"+file,"r") as fr:
                    word=fr.read().split()
                    for w in word:
                        w = w.lower().translate(None, "!@#$%^&*_-+=|\\;\/:.,<>?[]{}()")
                        if w  not in stop_words and w.isalpha():
                            word_vec[folder][w] += 1
                            file_vec[folder][key][w] += 1    
            key += 1                
    return word_vec, file_vec            

# train and create a model vector to be written to model file. Model vetor has probabilities of words lying
# in different topics
def train_model(visible, file_vec, count_topics):
    model_vector = collections.defaultdict(lambda: collections.defaultdict(lambda:0 ))
    for topic in visible:
        for file_index in visible[topic]:
            for word in file_vec[topic][file_index]:
                model_vector[topic][word] += file_vec[topic][file_index][word]
    for topic in model_vector:
        count_words = sum(model_vector[topic].values())
        for word in model_vector[topic]:
            model_vector[topic][word] = math.log (float(model_vector[topic][word]) / float(count_words))
    return model_vector
# Based on the fraction given we classify random files from each topic into two sections, visible and invisible. 
# Invisible labels cannot be seen and we label them based on the visible labels.
def segregate(fraction, count_topics)    :
    visible = collections.defaultdict(list)
    invisible = collections.defaultdict(list)        
    for t in count_topics:
        invisible[t] = list(range(count_topics[t]))
    for t in count_topics:
        loop_counter = int(math.ceil(fraction*count_topics[t]))
        for i in range(loop_counter):
            while len(visible[t]) != loop_counter:
                r = random.randint(0, count_topics[t])
                if r not in visible[t]:
                    visible[t].append(r)
                if r in invisible[t]:
                    invisible[t].remove(r)
    return visible, invisible        
# Using the trained model vector we classify invisible files into different topics and keep on appending the file 
# indices in visible so that they can be used in next iteration for training.
def classify_invisible(model_vector, invisible, visible, file_vec, count_topics):
    total_visible_files = sum(len(l) for l in visible)
    for topic in invisible:
        for file_index in invisible[topic]:
            prob = 0
            probability_topic = collections.defaultdict(lambda: collections.defaultdict(lambda:0 ))
            for word in file_vec[topic][file_index]:
                if word in model_vector[topic]:
                    prob += model_vector[topic][word]
                else:
                    model_vector[topic][word] = math.log(1E-9)    
                    prob += math.log(1E-9)    
            probability_topic[file_index][topic] = prob + math.log(float(len(visible[topic]))/float(total_visible_files))
    for file_index in probability_topic:
        topic = max(probability_topic[file_index], key = probability_topic[file_index].get)
        visible[topic].append(file_index)
        
#main function to parse all the arguments and iterate training and testing again to classify the invisible labels.
if __name__ == "__main__":
    mode=sys.argv[1]
    directory = sys.argv[2]
    model_file = sys.argv[3]
    if mode == "train":
        print "Train model. "
        fraction = float (sys.argv[4])
        if fraction == 0:
            fraction = bul;
        count_topics = create_count(directory)
        word_vec, file_vec = create_file_vec(directory)
        visible, invisible = segregate(fraction, count_topics)
        model_vector = train_model(visible, file_vec, count_topics)
        classify_invisible(model_vector, invisible, visible, file_vec, count_topics)
        for i in range(125):
            model_vector = train_model(visible, file_vec, count_topics)
            classify_invisible(model_vector, invisible, visible, file_vec, count_topics)
        total_visible_files = sum(len(l) for l in visible)
        for topic in visible:
            model_vector["CountPosterior"][topic] = len(visible[topic]) / total_visible_files
        distinctive_words = collections.defaultdict(list)
        for topic in model_vector:
            if topic != "CountPosterior":
                distinctive_words[topic] = sorted(model_vector[topic], key = lambda val : model_vector[topic][val], reverse = True)[:10]
        with open("distinctive.txt", 'w') as f:
            json.dump(distinctive_words, f, indent = 4)        
        with open(model_file, 'w') as f:
            json.dump(model_vector, f, indent = 4)
        print "Training done."    
    elif mode == "test":
        print "Testing started."
        model_vector = collections.defaultdict(lambda: collections.defaultdict(lambda:0 ))
        model_vector = json.load( open(model_file, 'r'))
        test(model_vector, directory)        
