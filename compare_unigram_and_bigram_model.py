# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:25:43 2016

@author: rmisra
"""
import numpy

vocab = open("vocab.txt",'r');
vocabulary = [line.strip('\n') for line in vocab];
vocab.close();

#sentence = "LAST WEEK THE STOCK MARKET FELL BY ONE HUNDRED POINTS."
sentence = "THE NINETEEN OFFICIALS SOLD FIRE INSURANCE."
sentence = sentence.strip('.').split(' ');

dictionary = {}
rev_dictionary = {}
for i in range(0,len(vocabulary)):
    dictionary[i+1] = vocabulary[i]
    rev_dictionary[vocabulary[i]] = i+1;

unigram = open("unigrams.txt",'r')
unigrams_frequency = [int(line.strip('\n')) for line in unigram]
unigram.close()

total_frequency = sum(unigrams_frequency)
probability_unigrams = numpy.zeros(len(sentence))

for i in range(0,len(sentence)):
    if(sentence[i] not in rev_dictionary):
        probability_unigrams[i] = float(unigrams_frequency[rev_dictionary['<UNK>']-1])/float(total_frequency);
    else:
        probability_unigrams[i] = float(unigrams_frequency[rev_dictionary[sentence[i]]-1])/float(total_frequency);

#print (unigrams_frequency[88])
log_likelihood_unigram = 0;
for i in range(0,len(sentence)):
    log_likelihood_unigram += numpy.log(probability_unigrams[i]);

print('log likelihood of unigram model on ' + ' '.join(sentence) + ' ->\t' + str(log_likelihood_unigram))


bigram = open("bigrams.txt",'r')
bigrams_frequency = [line.strip('\n').split('\t') for line in bigram]
bigram.close()

probability_bigrams = numpy.zeros((500,500))
for i in range(0,len(bigrams_frequency)):
    probability_bigrams[int(bigrams_frequency[i][0])-1][int(bigrams_frequency[i][1])-1] = float(bigrams_frequency[i][2])/unigrams_frequency[int(bigrams_frequency[i][0])-1]
    
#print (' P ' + str(probability_bigrams[26][0]))
log_likelihood_bigram = 0;
if(sentence[0] not in rev_dictionary):
    log_likelihood_bigram += numpy.log(probability_bigrams[rev_dictionary['<s>']-1][rev_dictionary['<UNK>']-1]);
else:
    log_likelihood_bigram += numpy.log(probability_bigrams[rev_dictionary['<s>']-1][rev_dictionary[sentence[0]]-1]);
                                           

for i in range(1,len(sentence)):
    if(sentence[i] not in rev_dictionary and sentence[i-1] not in rev_dictionary):
        log_likelihood_bigram += numpy.log(probability_bigrams[rev_dictionary['<UNK>']-1][rev_dictionary['<UNK>']-1]);
    elif (sentence[i] not in rev_dictionary):
        log_likelihood_bigram += numpy.log(probability_bigrams[rev_dictionary[sentence[i-1]]-1][rev_dictionary['<UNK>']-1]);
    elif (sentence[i-1] not in rev_dictionary):
        log_likelihood_bigram += numpy.log(probability_bigrams[rev_dictionary['<UNK>']-1][rev_dictionary[sentence[i]]-1]);
    else:
        log_likelihood_bigram += numpy.log(probability_bigrams[rev_dictionary[sentence[i-1]]-1][rev_dictionary[sentence[i]]-1]);
                                        
    print(log_likelihood_bigram);
                                           
    #print(log_likelihood_bigram)

print('log likelihood of bigram model on ' + ' '.join(sentence) + ' ->\t' + str(log_likelihood_bigram))
