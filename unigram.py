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
for i in range(0,len(vocabulary)):
    dictionary[i+1] = vocabulary[i]

unigram = open("unigrams.txt",'r')
unigrams_frequency = [int(line.strip('\n')) for line in unigram]
unigram.close()

total_frequency = sum(unigrams_frequency)
probability_unigrams = numpy.zeros(len(vocabulary))

for i in range(0,len(vocabulary)):
    probability_unigrams[i] = unigrams_frequency[i]/total_frequency;

for i in range(0,len(vocabulary)):
    if(dictionary[i+1][0]=='A'):
        print('ML Probability of ' + dictionary[i+1] + ' ->\t' + str(probability_unigrams[i]))
