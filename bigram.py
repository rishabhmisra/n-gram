# This is a general implementation of predicting words using the bigram model
# Input: One of the valid tokens from the vocabulary
# Output: Top-5 most likely tokens that can follow the input token
 
"""
Created on Sat Oct 22 12:25:43 2016
@author: rmisra
"""

import numpy

# importing the vocabulary
vocab = open("vocab.txt",'r');
vocabulary = [line.strip('\n') for line in vocab];
vocab.close();

# data structures to link tokens with their indices
dictionary = {}
rev_dictionary = {}
for i in range(0,len(vocabulary)):
    dictionary[i+1] = vocabulary[i]
    rev_dictionary[vocabulary[i]] = i+1;

# importing the unigram frequencies
unigram = open("unigrams.txt",'r')
unigrams_frequency = [int(line.strip('\n')) for line in unigram]
unigram.close()

# importing the bigram frequencies
bigram = open("bigrams.txt",'r')
bigrams_frequency = [line.strip('\n').split('\t') for line in bigram]
bigram.close()

# setting the current word to the token "THE"
w1 = input("Input a valid token from the vocabulary\n");   #THE

# Likelihood of words following "THE"
P_w2_given_w1 = numpy.zeros(len(vocabulary));

# Likelihood of words following any other token
probability_bigrams = numpy.zeros(len(bigrams_frequency))

# going over all the bigram entries
for i in range(0,len(probability_bigrams)):
    # calculating ML probability for each pair
    probability_bigrams[i] = float(bigrams_frequency[i][2])/float(unigrams_frequency[int(bigrams_frequency[i][0])-1])
    # if the first column of the entry correspond to the input token, record it
    if(int(bigrams_frequency[i][0]) == rev_dictionary[w1]):
        P_w2_given_w1[int(bigrams_frequency[i][1])-1] = probability_bigrams[i]

# find top-5 tokens
P_max_w2_given_w1 = numpy.argpartition(P_w2_given_w1, -5)[-5:]
P_max_w2_given_w1 = P_max_w2_given_w1[numpy.argsort(P_w2_given_w1[P_max_w2_given_w1])]

# print top-5 tokens
for i in range(0,len(P_max_w2_given_w1)):
    print('ML Probability of word ' + w1 + ' followed by ' + dictionary[P_max_w2_given_w1[i]+1] + ' ->\t' + str(P_w2_given_w1[P_max_w2_given_w1[i]]))
