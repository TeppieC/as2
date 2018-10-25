from __future__ import division  # floating point division
import csv
import numpy as np

import regressionalgorithms as algs
import matplotlib
import matplotlib.pyplot as plt

def plot_comparison(*args):
	'''
	Plot the SGD(step-size=0.01), SGD(step-size=0.001) and BatchGD in the same graph

	Note: it is a helper function. Only used to plot the comparison plot between SGD and BatchGD
	'''
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))

	for i in range(len(args)):
		with open(args[i],'r') as f:
			data = f.readlines()
		epoch_list = []
		runtime_list = []
		error_list = []
		for line in data:
			epoch, runtime, error= line.strip().split(',')
			epoch_list.append(int(epoch)+1)
			runtime_list.append(float(runtime))
			error_list.append(float(error))
		ax[0].plot(runtime_list, error_list, 'C%d'%i, label=str(i))
		ax[1].plot(epoch_list, error_list, 'C%d'%i, label=str(i))
		
	

	ax[0].set(xlabel='runtime (ms)', ylabel='cost',
				title='Cost v.s Rumtime')
	ax[1].set(xlabel='epoch', ylabel='cost',
				title='Cost v.s Epoch')
	ax[0].grid()
	ax[1].grid()
	
	fig.savefig("comparison.png") # plot will be saved to the current directory
	plt.show()

if __name__ == '__main__':
	plot_comparison('SGD-0.01-cost.csv','SGD-0.001-cost.csv','BGD-cost.csv')
