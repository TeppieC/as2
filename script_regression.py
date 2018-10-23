from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import utilities as utils

import dataloader as dtl
import regressionalgorithms as algs
import matplotlib
import matplotlib.pyplot as plt

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numruns = 1

    regressionalgs = {
                #'Random': algs.Regressor(),
                #'Mean': algs.MeanPredictor(),
                #'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                #'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                #'FSLinearRegression50': algs.FSLinearRegression({'features': range(384)}),
                #'RidgeLinearRegression': algs.RidgeLinearRegression({'regwgt': 0.01}),
                #'LassoRegression': algs.LassoLinearRegression({'regwgt': 0.01}),
                'SGD': algs.SGDLinearRegression({'num_epoch':1000, 'stepsize':0.01}),
                'SGD': algs.SGDLinearRegression({'num_epoch':1000, 'stepsize':0.001}),
                'BGD': algs.BatchGDLinearRegression({'num_epoch':1000}),                
                #'MiniGD': algs.MiniBatchGDLinearRegression({'num_epoch':1000, 'batch_size':10, 'stepsize':0.01}),
                #'SGDLinearRegressionRmsprop': algs.SGDLinearRegressionRmsprop({'num_epoch':1000, 'stepsize':0.001, 'decay':0.9}),
                #'SGDLinearRegressionAmsgrad': algs.SGDLinearRegressionAmsgrad({'num_epoch':1000, 'stepsize':0.001, 'beta1':0.9,'beta2':0.99}),

             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    # NOTE: this is lambda
    parameters = (
        {'regwgt': 0.01},
        #{'regwgt': 0.1},
        #{'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0]) #testset[0]: test data Xtest, testset[1]: ground truth Ytest
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error

    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:]) #start case
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:]) #extract avg error for parameter p of this learning algorithm
            # there are multiple runs, aveerror is the mean of the all runs
            #print('aveerror for ',learnername, ":",aveerror,", under parameter:",parameters[p])
            utils.stdev(errors[learnername][p,:])/np.sqrt(numruns)

            if aveerror < besterror:
                besterror = aveerror
                bestparams = p #substitue the best parameter which generate the minimum error

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print('bestparams for',learnername,":",parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))

