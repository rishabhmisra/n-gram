# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:25:43 2016

@author: rmisra
"""
import numpy

vocab = open("vocab.txt",'r');
vocabulary = [line.strip('\n') for line in vocab];
vocab.close();

dictionary = {}
rev_dictionary = {}
for i in range(0,len(vocabulary)):
    dictionary[i+1] = vocabulary[i]
    rev_dictionary[vocabulary[i]] = i+1;

unigram = open("unigrams.txt",'r')
unigrams_frequency = [int(line.strip('\n')) for line in unigram]
unigram.close()

bigram = open("bigrams.txt",'r')
bigrams_frequency = [line.strip('\n').split('\t') for line in bigram]
bigram.close()

w1 = 'THE'
P_w2_given_w1 = numpy.zeros(len(vocabulary));

probability_bigrams = numpy.zeros(len(bigrams_frequency))
for i in range(0,len(probability_bigrams)):
    probability_bigrams[i] = float(bigrams_frequency[i][2])/float(unigrams_frequency[int(bigrams_frequency[i][0])-1])
    if(int(bigrams_frequency[i][0]) == rev_dictionary[w1]):
        P_w2_given_w1[int(bigrams_frequency[i][1])-1] = probability_bigrams[i]

P_max_w2_given_w1 = numpy.argpartition(P_w2_given_w1, -5)[-5:]
P_max_w2_given_w1 = P_max_w2_given_w1[numpy.argsort(P_w2_given_w1[P_max_w2_given_w1])]

for i in range(0,len(P_max_w2_given_w1)):
    print('ML Probability of word ' + w1 + ' followed by ' + dictionary[P_max_w2_given_w1[i]+1] + ' ->\t' + str(P_w2_given_w1[P_max_w2_given_w1[i]]))
