from __future__ import division  # floating point division
import numpy as np
import math
import time
import utilities as utils

import matplotlib
import matplotlib.pyplot as plt


def write_to_file(filename, runtime_list, error_list, num_epoch):
    '''
    Helper function to write data to file, in order to plot the curve
    '''
    with open(filename,'w') as f:
        for i in range(num_epoch):
            f.write(str(i)+','+str(runtime_list[i])+','+str(error_list[i])+'\n')


class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        numfeatures = Xtrain.shape[1]
        penalty = np.multiply(self.params['regwgt'],np.identity(numfeatures))
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples, penalty)), Xtrain.T),ytrain)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class LassoLinearRegression(Regressor):
    """docstring for LassoRegression"""
    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)
        self.max_runs = 1000000

    def proximal(self, w):
        i = 0
        while i < w.shape[0]:
            temp = np.dot(self.stepsize,self.params['regwgt'])
            if w[i]>temp:
                w[i]-=temp
            elif w[i]<-temp:
                w[i]+=temp
            else:
                w[i] = 0
            i+=1
        return w

    def cost(self, w, X, y):
        penalty = np.multiply(self.params['regwgt'],np.linalg.norm(w,ord=1))
        XWminusY = np.subtract(np.dot(X,w),y)
        return ((np.dot(XWminusY.T, XWminusY)+penalty)/(2*X.shape[0])).item()

    def learn(self, X, y):
        numsamples = X.shape[0]
        numfeatures = X.shape[1]
        y = y.reshape(numsamples,1) #reshape y to numsamples*1

        w = np.zeros(numfeatures).reshape(numfeatures,1)#reshape w to numfeatures*1
        err = np.Infinity
        tolerance = 10*np.exp(-4)
        xx = np.dot(X.T,X)/numsamples
        xy = np.dot(X.T,y)/numsamples
        self.stepsize = 1/(2*np.linalg.norm(xx,ord=2))

        cost_w = self.cost(w, X, y)
        runs = 0

        while np.abs(cost_w - err)>tolerance and not runs>self.max_runs:
            err = cost_w
            w = self.proximal(w-self.stepsize*np.dot(xx,w)+self.stepsize*xy) # update w
            cost_w = self.cost(w, X, y)
            runs+=1
        self.weights = w
        

    def predict(self, Xtest):
        ## Modified
        ytest = np.dot(Xtest, self.weights).reshape(Xtest.shape[0])
        return ytest

    
class SGDLinearRegression(Regressor):
    '''
    e) Report the error, for a step-size of 0.01 and 1000 epochs
    '''
    '''
    f) 
    Compare stochastic gradient descent to batch gradient descent, in terms of the number of times the entire
    training set is processed

    Set the step-size to 0.01 for stochastic gradient descent. 
    Report the error versus epochs, where one epoch involves processing the training set once. 
    Report the error versus runtime
    '''
    def __init__(self, parameters={}):
        self.params = {'num_epoch':1000, 'stepsize':0.01}
        self.reset(parameters)

    def learn(self, X, y):
        numsamples = X.shape[0]
        numfeatures = X.shape[1]
        y = y.reshape(numsamples,1) #reshape y to numsamples*1
        w = np.ones(numfeatures).reshape(numfeatures,1) # numfeatures*1
        # concatenate the training X and y together, before shuffling
        data = np.concatenate((X,y), axis=1) # numsamples*(numfeatures+1)

        #errors = []
        #times = []
        start = time.time()
        for i in range(self.params['num_epoch']):
            np.random.shuffle(data)
            for j in range(numsamples):
                # extract one data point (xj.T, yj)
                XjT = (data.T[:-1]).T[j].reshape(1,numfeatures) #1*numfeatures
                yj = (data.T[-1:]).T[j].reshape(1,1) #1*1 
                # compute gradient
                g = np.multiply((np.dot(XjT, w)-yj),XjT.T).reshape(numfeatures,1)#numfeatures*1
                # update weights
                w = w - np.dot(self.params['stepsize'],g)

            # collect misc info to draw the plot
            elapsed_time = time.time()-start
            print('epoch',i+1,'| cost SGD:', np.linalg.norm(np.dot(X,w)-y)**2/(2*numsamples),'| L2 error:',np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples, '| runtime:',elapsed_time)
            #times.append(elapsed_time)
            #errors.append(np.linalg.norm(np.dot(X,w)-y)**2/(2*numsamples))
            #errors.append(np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples)

        '''
        # Draw plot
        #   plot 1: L2 error v.s runtime
        #   plot 2: L2 error v.s epoch
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
        ax[0].plot(times, errors)
        ax[0].set(xlabel='runtime (ms)', ylabel='cost',
                   title='SGD, %d epoches, step-size=%.4f'%(self.params['num_epoch'], self.params['stepsize']))
        ax[0].grid()
        ax[1].plot(range(self.params['num_epoch']), errors)
        ax[1].set(xlabel='epoch', ylabel='cost',
                   title='SGD, %d epoches, step-size=%.4f'%(self.params['num_epoch'], self.params['stepsize']))
        ax[1].grid()
        fig.savefig("sgd.png") # plot will be saved to the current directory
        plt.show()
        '''

        # write the performance data to file
        #write_to_file('SGD.csv', times, errors, self.params['num_epoch'])

        #print('final cost SGD:', np.linalg.norm(np.dot(X,w)-y)**2/(2*numsamples))
        
        self.weights = w
        

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights).reshape(Xtest.shape[0])
        return ytest

    
class BatchGDLinearRegression(Regressor):
    '''
    Compare stochastic gradient descent to batch gradient descent, in terms of the number of times the entire
    training set is processed

    Report the error versus epochs, where one epoch involves processing the training set once. 
    Report the error versus runtime


    NOTE: As instructed by TA's in lab, this algorithm is implemented in a slightly different way as it was on class notes.
    The Batch Gradient Descent ends after 1000 epoches, even though it usually takes ~100 epoches to converge.
    In class notes, the Batch GD ends if the cost is no longer descreasing within a threshold.
    '''
    def __init__(self, parameters={}):
        self.params = {'num_epoch':1000}
        self.reset(parameters)

    def cost(self, w, X, y):
        return np.linalg.norm(np.dot(X,w)-y)**2/(2*X.shape[0])

    def deriv_cost(self, w, X, y):
        return np.dot(X.T,np.dot(X,w)-y)/X.shape[0]

    def learn(self, X, y):
        # X numsamples*numfeatures
        numsamples = X.shape[0]
        numfeatures = X.shape[1]
        y = y.reshape(numsamples,1) #reshape y to numsamples*1
        w = np.random.rand(numfeatures).reshape(numfeatures,1) # numfeatures*1

        #times = []
        #errors = []
        cost = np.linalg.norm(np.dot(X,w)-y)**2/(2*numsamples)
        start = time.time()
        for i in range(self.params['num_epoch']):
            # compute gradient
            g = self.deriv_cost(w, X, y) 
            # use backtracking line search to find the step-size
            w, a = self.line_search(w, X, y, cost, g) 
            # compute loss
            cost = self.cost(w, X, y)

            # collect misc data to draw the plot
            elapsed_time = time.time()-start
            print('epochs',i+1,'| cost Batch GD:', cost,'| L2 error:',np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples, '| runtime:',elapsed_time)
            #times.append(elapsed_time)
            #errors.append(cost)
            #errors.append(np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples)
        
        '''
        # Draw plot
        #   plot 1: L2 error v.s runtime
        #   plot 2: L2 error v.s epoch
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
        ax[0].plot(times, errors)
        ax[0].set(xlabel='runtime (ms)', ylabel='cost',
                   title='Batch GD, %d epoches'%self.params['num_epoch'])
        ax[0].grid()
        ax[1].plot(range(self.params['num_epoch']), errors)
        ax[1].set(xlabel='epoch', ylabel='cost',
                   title='Batch GD, %d epoches'%self.params['num_epoch'])
        ax[1].grid()
        fig.savefig("batch-gd.png") #plot will be save to current directory
        plt.show()
        '''

        # write performace data to file
        #write_to_file('BGD.csv', times, errors, self.params['num_epoch'])

        
        #print('final cost Batch GD with line search:',cost)
        self.weights = w
        
    def line_search(self, wt, X, y, obj, g):
        a = 0.01
        tolerance = 10*np.exp(-4)
        max_iter = 100
        decay = 0.7
        
        w = wt
        i = 0
        while i<max_iter:
            w = wt - a*g
            if self.cost(w, X, y) < obj-tolerance: break
            a*=decay
            i+=1

        if i == max_iter:
            return (wt, 0)
        else:
            return (w, a)

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights).reshape(Xtest.shape[0])
        return ytest




class SGDLinearRegressionRmsprop(Regressor):
    '''
    e) Report the error, for a step-size of 0.01 and 1000 epochs
    '''
    '''
    f) 
    Compare stochastic gradient descent to batch gradient descent, in terms of the number of times the entire
    training set is processed

    Set the step-size to 0.01 for stochastic gradient descent. 
    Report the error versus epochs, where one epoch involves processing the training set once. 
    Report the error versus runtime
    '''
    def __init__(self, parameters={}):
        self.params = {'num_epoch':1000, 'stepsize':0.001, 'decay':0.9}
        self.reset(parameters)

    def learn(self, X, y):
        # X numsamples*numfeatures
        numsamples = X.shape[0]
        numfeatures = X.shape[1]
        y = y.reshape(numsamples,1) #reshape y to numsamples*1
        w = np.ones(numfeatures).reshape(numfeatures,1) # numfeatures*1
        eps = np.exp(-8)
        # concatenate X and y before shuffling
        data = np.concatenate((X,y), axis=1) # numsamples*(numfeatures+1)

        #initialize param for rmsprop
        ms = np.ones(numfeatures).reshape(numfeatures,1)

        #errors = [] # for plotting
        for i in range(self.params['num_epoch']):
            np.random.shuffle(data)
            for j in range(numsamples):
                # extract a single data point (xj,yj)
                XjT = (data.T[:-1]).T[j].reshape(1,numfeatures) #1*numfeatures
                yj = (data.T[-1:]).T[j].reshape(1,1) #1*1 
                # compute gradient
                g = np.multiply((np.dot(XjT, w)-yj),XjT.T).reshape(numfeatures,1)#numfeatures*1

                # compute meansquare
                ms = self.meansquare(ms, g, self.params['decay'])
                # update weights with a decaying rmsprop learning rate
                w = w - np.dot(self.params['stepsize'], g/np.sqrt(ms+eps))
                
            #print('epoch',i,'| cost on SGD rmsprop:', np.linalg.norm(np.dot(X,w)-y)**2/(2*numsamples)) #for debugging
            #errors.append(np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples) # for plotting
        
        '''
        # draw graph
        fig, ax = plt.subplots()
        ax.plot(range(self.params['num_epoch']), errors)
        ax.set(xlabel='epoch', ylabel='L2 error',
                   title='SGD-rmsprop, 1000 epoches, step-size=%f'%self.params['stepsize'])
        ax.grid()
        fig.savefig("SGD-rmsprop.png")
        plt.show()
        '''
        #print('After %d epochs, L2-error of SGD rmsprop (w.r.t training set):'%self.params['num_epoch'], np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples)

        self.weights = w
        

    def meansquare(self,ms,g,decay):
        return decay*ms+(1-decay)*np.power(g,2)


    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights).reshape(Xtest.shape[0])
        return ytest


class SGDLinearRegressionAmsgrad(Regressor):
    '''
    e) Report the error, for a step-size of 0.01 and 1000 epochs
    '''
    '''
    f) 
    Compare stochastic gradient descent to batch gradient descent, in terms of the number of times the entire
    training set is processed

    Set the step-size to 0.01 for stochastic gradient descent. 
    Report the error versus epochs, where one epoch involves processing the training set once. 
    Report the error versus runtime
    '''
    def __init__(self, parameters={}):
        self.params = {'num_epoch':1000, 'stepsize':0.001, 'beta1':0.9,'beta2':0.99}
        self.reset(parameters)

    def learn(self, X, y):
        # X numsamples*numfeatures
        numsamples = X.shape[0]
        numfeatures = X.shape[1]
        y = y.reshape(numsamples,1) #reshape y to numsamples*1
        w = np.ones(numfeatures).reshape(numfeatures,1) # numfeatures*1
        data = np.concatenate((X,y), axis=1) # numsamples*(numfeatures+1)

        # initialize params for Amsgrad
        m = np.zeros(numfeatures).reshape(numfeatures,1)
        v = np.zeros(numfeatures).reshape(numfeatures,1)
        vhat = np.zeros(numfeatures).reshape(numfeatures,1)
        eps = np.exp(-8)

        #errors = [] #for plotting
        for i in range(self.params['num_epoch']):
            np.random.shuffle(data)
            for j in range(numsamples):
                # extract one single data point (xj, yj)
                XjT = (data.T[:-1]).T[j].reshape(1,numfeatures) #1*numfeatures
                yj = (data.T[-1:]).T[j].reshape(1,1) #1*1 
                # compute gradient
                g = np.multiply((np.dot(XjT, w)-yj),XjT.T).reshape(numfeatures,1)#numfeatures*1

                # Amsgrad
                m = np.dot(self.params['beta1'], m)+np.dot(1-self.params['beta1'], g)
                v = np.dot(self.params['beta2'], v)+np.dot(1-self.params['beta2'], np.power(g,2))
                vhat = np.maximum(vhat,v)

                # update w
                w = w - np.dot(self.params['stepsize'], m/(np.sqrt(vhat)+eps))

            #errors.append(np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples)#for plotting
        '''
        
        fig, ax = plt.subplots()
        ax.plot(range(self.params['num_epoch']), errors)
        ax.set(xlabel='epoch', ylabel='L2 error',
                   title='SGD-amsgrad, 1000 epoches, step-size=%f'%self.params['stepsize'])
        ax.grid()
        fig.savefig("sgd-amsgrad.png")
        plt.show()
        '''
        #print('After %d epochs, L2-error of SGD amsgrad (w.r.t training set):'%self.params['num_epoch'], np.linalg.norm(np.subtract(np.dot(X,w),y))/numsamples)
        
        self.weights = w
        

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights).reshape(Xtest.shape[0])
        return ytest