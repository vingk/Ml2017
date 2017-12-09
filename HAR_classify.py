
"""
Author: Vinaykumar Kulkarni
M.Sc, Univeristy of Alberta.

In this project, we try to predict Human activity from smart phone data 
We compare Naive Bayes, Dicision tree and Neural Network based learning methods and use accuracy as a performance metric.
We report classification report with precision recall and f1 score and use statistical significance t test
to test if the findings are statistically significant.
--
Project completed as part of Machine Learning Course CMPUT551( Fall 2017), under Prof. Martha White.
December 8th, 2017
"""

from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from sklearn.metrics import classification_report    #for classification f1 score 
from sklearn.naive_bayes import GaussianNB   #Naive bayes implementation 
from sklearn.tree import DecisionTreeClassifier   #CART implementation
from sklearn.neural_network import MLPClassifier   #Neural Networks
from sklearn.model_selection import GridSearchCV    #Grid search cross validation with k fold
from scipy.stats import ttest_ind  #T test


#Load the dataset from txt file
def loadcsv(filename):
    dataset = np.genfromtxt(filename,dtype=float, delimiter='')
    return dataset
   

#main program
if __name__ == '__main__':

    #choice of algorithms
    classalgs = {
                 'NaiveBayes': GaussianNB(),
                 'DecisionTree': DecisionTreeClassifier(),
                 'NeuralNetworks': MLPClassifier()
                }
    numalgs = len(classalgs)

    #Load the data from different files and combine them to form (X,y) dataset
    X=np.concatenate((loadcsv('train/X_train.txt'),loadcsv('test/X_test.txt')))
    y = np.concatenate((loadcsv('train/y_train.txt'),loadcsv('test/y_test.txt')))

    #check if data is loaded properly
    #print X.shape
    #print y.shape
    
    numruns =10  #specify the number of runs
    numsamples = X.shape[0]

    #Store errors for each run for every algorithm.
    besterrors = {}
    for learnername in classalgs:
        besterrors[learnername] = np.zeros(numruns)

    #parameters for all the algorithms.
    params = {'DecisionTree':{'class_weight':['balanced'],'max_depth':[None,X.shape[1],int(math.log(X.shape[0]))],'min_samples_split':[2,4,8,16,32],'min_impurity_decrease':[0.0,0.1,0.01,0.001,0.0001]},
              'NeuralNetworks':{'alpha':[0.01,0.001,0.0001,0.00001],'hidden_layer_sizes':[(5,),(14,),(6,),(561,),(283,),(100,)]},
               'NaiveBayes':{}
               }

    #itarate for each run
    for run in range(numruns):

        print 'Run number: '+str(run)+' '

        #Reshuffle the Dataset.
        rng_state = np.random.get_state() # get the state of random generator
        np.random.shuffle(X) # shuffle Xtrain values
        np.random.set_state(rng_state)  # reset random generator to original state
        np.random.shuffle(y) # shuffle Ytrain values

        #split the data into 70% train and 30% test
        Xtrain = X[:int(numsamples*0.70), :]
        ytrain = y[:int(numsamples*0.70)]
        Xtest = X[int(numsamples*0.70): int(numsamples), :]
        ytest = y[int(numsamples*0.70): int(numsamples)]

        #Train and Test for each algorithm.
        for learnername, learner in classalgs.items():
            
            print learnername

            #Perform Grid search cross validation to find the best parameter
            clf = GridSearchCV(estimator=learner, param_grid=params[learnername], n_jobs=100, cv=10)
            #Train the model
            clf.fit(Xtrain,ytrain)
            #Best parameter found during cross validation
            print clf.best_params_
            #best training accuracy found
            print clf.best_score_
            #store the best error for each algorithm
            besterrors[learnername][run] =   clf.score(Xtest,ytest)   #Testing set accuracy
            print besterrors[learnername][run]
            #print f score , precision and recall over each class/
            print classification_report(ytest,clf.predict(Xtest))

    #Print all the errors over runs for each algorithm  
    for learnername in classalgs:
        print 'Errors across '+str(numruns)+' runs for '+learnername+''
        print besterrors[learnername]

        
    print 'Performing statitical tests on the data'
    
    print 'T test for Neural Network & Naive Bayes'
    print ttest_ind(besterrors['NaiveBayes'],besterrors['NeuralNetworks'],equal_var=False)
    
    print 'T test for Neural Network & Decision tree'
    print ttest_ind(besterrors['DecisionTree'],besterrors['NeuralNetworks'],equal_var=False)
  
    print 'T test for Naive Bayes & Decision Tree'
    print ttest_ind(besterrors['NaiveBayes'],besterrors['DecisionTree'],equal_var=False)
  
