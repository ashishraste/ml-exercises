#!/usr/bin/python

import numpy as np
import os
from scipy.io import loadmat

X = Y = None
numSamples = None
theta1 = theta2 = None

# dataset path
curDir = os.getcwd()
datasetPath = curDir + '/datasets/ex3data1.mat'
weightsFile = curDir + '/datasets/ex3weights.mat'


def loadData():
  '''
  Loads the dataset, initializes the input-samples' and output-labels' matrices.
  '''
  global X,Y,numSamples,theta
  mat = loadmat(datasetPath)
  X = mat['X']; Y = mat['y']
  numSamples = X.shape[0]
  X = np.c_[np.ones((numSamples,1)),X]
  # print X.shape


def loadWeights():
	"""
	@brief      Loads pre-trained weights.	
	"""
	global theta1, theta2
	mat = loadmat(weightsFile)
	theta1 = mat['Theta1']
	theta2 = mat['Theta2']
	# print theta1.shape, theta2.shape


def sigmoid(Z):
  ''' 
  Returns the sigmoid function values of the given input array. 
  '''
  return (1 / (1 + np.exp(-Z)))


def predictAccuracy(theta1, theta2, X):
	"""
	@brief      Predicts the accuracy of trained parameters (theta-matrices).
	
	@param      theta1  parameters mapping layer 1 to layer 2 of the neural-network.
	@param      theta2  parameters mapping layer 2 to layer 3 of the neural-network.
	@param      X       Samples from the dataset, each of them described by their pixel values (400 in number).
	
	@return     Prediction of the numbers.
	"""
	z2 = theta1.dot(X.T)
	a2 = sigmoid(z2)
	a2 = np.r_[np.ones((1,numSamples)),a2]
	z3 = theta2.dot(a2)
	a3 = sigmoid(z3)  # shape 10x5000
	return np.argmax(a3.T,axis=1)+1  # return the predicted values for each of the 5000 samples


if __name__=="__main__":
	loadData()
	loadWeights()
	prediction = predictAccuracy(theta1, theta2, X)
	print 'training accuracy = {}%'.format(np.mean(prediction == Y.ravel())*100)