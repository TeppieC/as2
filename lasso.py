from __future__ import division  # floating point division
import numpy as np
import math
import csv


import dataloader as dtl
import utilities as utils
#from regressionalgorithms import Regression

class LassoRegression():
	"""docstring for LassoRegression"""
	def __init__(self, regwgt, stepsize=0, max_runs=1000):
		self.regwgt = regwgt
		self.stepsize = stepsize
		self.max_runs = max_runs

	def proximal(self, w):
		i = 0
		while i < w.shape[0]:
			temp = self.stepsize*self.regwgt
			if w[i]>temp:
				w[i]-=temp
			elif w[i]<-temp:
				w[i]+=temp
			else:
				w[i] = 0
			i+=1
		return w

	def cost(self, w, X, y):
		penalty = np.multiply(self.regwgt,np.linalg.norm(w,ord=1))
		XWminusY = np.subtract(np.dot(X,w),y)
		return (np.dot(XWminusY.T, XWminusY)+penalty).item()

	def batch_gd(self, X, y):
		numsamples = X.shape[0]
		numfeatures = X.shape[1]
		y = y.reshape(numsamples,1) #reshape y to numsamples*1

		w = np.zeros(numfeatures).reshape(numfeatures,1)
		err = np.Infinity
		tolerance = 10*np.exp(-4)
		xx = np.dot(X.T,X)/numsamples
		xy = np.dot(X.T,y)/numsamples
		self.stepsize = 1/(2*np.linalg.norm(xx, ord=1))

		print('start GD')
		cost_w = self.cost(w, X, y)
		runs = 0

		while np.abs(cost_w - err)>tolerance and not runs>self.max_runs:
			err = cost_w
			#w = np.array([self.proximal(wi-self.stepsize*np.dot(xx,wi)+self.stepsize*xy) for wi in w]).reshape(numfeatures, 1) #update w
			w = self.proximal(w-self.stepsize*np.dot(xx,w)+self.stepsize*xy) #update w
			cost_w = self.cost(w, X, y)
			#print('descent, now cost:', cost_w)
			runs+=1
		print('end GD',runs,'descents to reach optimal sparse solution')
		self.weights = w
		return w

	def predict(self, Xtest):
		ytest = np.dot(Xtest, self.weights)
		return ytest

'''
from script_regression.py
'''

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

	# Enable the best parameter to be selected, to enable comparison
	# between algorithms with their best parameter settings
	# NOTE: this regwgt is lambda
	parameters = (0.01,0.1,1.0)
	numparams = len(parameters)

	errors = {}
	errors["Lasso"] = np.zeros((numparams,numruns))

	for r in range(numruns):
		trainset, testset = dtl.load_ctscan(trainsize,testsize)
		print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

		for p in range(numparams):
			regwgt = parameters[p]
			learner = LassoRegression(regwgt,0,10000)

			print ('Running learner = Lasso on parameters', regwgt)
			# Train model
			best_w = learner.batch_gd(trainset[0], trainset[1])
			# Test model
			predictions = learner.predict(testset[0])
			error = geterror(testset[1], predictions)
			print ('Error for lasso: ' + str(error))
			errors["Lasso"][p,r] = error


	besterror = np.mean(errors["Lasso"][0,:])
	bestparams = 0
	for p in range(numparams):
		aveerror = np.mean(errors["Lasso"][p,:])
		if aveerror < besterror:
			besterror = aveerror
			bestparams = p

	# Extract best parameters
	#learner.reset(parameters[bestparams])
	#print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
	print ('Average error for Lasso :'  + str(besterror))

	