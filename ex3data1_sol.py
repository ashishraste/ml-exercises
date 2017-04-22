#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import scipy.io

curDir = os.getcwd()
datasetPath = curDir + '/datasets/ex3/ex3data1.mat'

X = X1 = Y = None
theta = None
numFeatures = None
numSamples = None


def visualizeData(Xrandom):
	fig = plt.figure()
	plt.gray()
	width = height = 20  # in pixels
	for index in Xrandom:
		print index
		sample = X[index]
		print Y[index] 
		plt.imshow(sample.reshape(width,height))
		plt.show()


def loadData():
	global X,Y,numSamples,numFeatures
	mat = scipy.io.loadmat(datasetPath)
	X = mat['X']
	Y = mat['y']
	numSamples,numFeatures = X.shape


if __name__=='__main__':
	loadData()
	Xr = np.random.randint(1,5000,2)  # display 2 random images within the dataset
	visualizeData(Xr)
