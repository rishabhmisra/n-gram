"""
Created on Sat Oct 22 12:25:43 2016

@author: rmisra
"""
import numpy
import matplotlib.pylab as py

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


bigram = open("bigrams.txt",'r')
bigrams_frequency = [line.strip('\n').split('\t') for line in bigram]
bigram.close()

probability_bigrams = numpy.zeros((500,500))
for i in range(0,len(bigrams_frequency)):
    probability_bigrams[int(bigrams_frequency[i][0])-1][int(bigrams_frequency[i][1])-1] = float(bigrams_frequency[i][2])/unigrams_frequency[int(bigrams_frequency[i][0])-1]
   
plt = numpy.zeros(101)
l = numpy.zeros(101)
for j in range(0,101):
    lamda = float(j)/100.0;
    log_likelihood_mixture = 0;
    if(sentence[0] not in rev_dictionary):
        log_likelihood_mixture += numpy.log((1-lamda)*float(unigrams_frequency[rev_dictionary['<UNK>']-1])/float(total_frequency) + lamda*probability_bigrams[rev_dictionary['<s>']-1][rev_dictionary['<UNK>']-1]);
    else:
        log_likelihood_mixture += numpy.log((1-lamda)*float(unigrams_frequency[rev_dictionary[sentence[0]]-1])/float(total_frequency) + lamda*probability_bigrams[rev_dictionary['<s>']-1][rev_dictionary[sentence[0]]-1]);
    
    for i in range(1,len(sentence)):
        if(sentence[i] not in rev_dictionary and sentence[i-1] not in rev_dictionary):
            log_likelihood_mixture += numpy.log(((1-lamda)*float(unigrams_frequency[rev_dictionary['<UNK>']-1])/float(total_frequency)) + lamda*probability_bigrams[rev_dictionary['<UNK>']-1][rev_dictionary['<UNK>']-1]);
        elif (sentence[i] not in rev_dictionary):
            log_likelihood_mixture += numpy.log(((1-lamda)*float(unigrams_frequency[rev_dictionary['<UNK>']-1])/float(total_frequency)) + lamda*probability_bigrams[rev_dictionary[sentence[i-1]]-1][rev_dictionary['<UNK>']-1]);
        elif (sentence[i-1] not in rev_dictionary):
            log_likelihood_mixture += numpy.log(((1-lamda)*float(unigrams_frequency[rev_dictionary[sentence[i]]-1])/float(total_frequency)) + lamda*probability_bigrams[rev_dictionary['<UNK>']-1][rev_dictionary[sentence[i]]-1]);
        else:
            log_likelihood_mixture += numpy.log(((1-lamda)*float(unigrams_frequency[rev_dictionary[sentence[i]]-1])/float(total_frequency)) + lamda*probability_bigrams[rev_dictionary[sentence[i-1]]-1][rev_dictionary[sentence[i]]-1]);
                                                 
    #print (str(j) + ' -> ' +str(log_likelihood_mixture))                                            
    plt[j] = log_likelihood_mixture;
    l[j] = lamda;

py.plot(l, plt, 'r-')
py.tick_params(labelright = True)
py.xlabel('Lambda')
py.ylabel('Log Likelihood')
py.title("Lambda vs Log Likelihood")
py.savefig("Log Likelihood Mixture.pdf", bbox_inches='tight')
py.show()

''' lamda 0.41 -> -39.9536799305 '''