#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import os

# Steps
# 1. Load the data, observed samples' shape is [numSamples x numFeatures]. 
# 2. Initialize parameteric vector theta whose shape is [numFeatures x 1].
# 3. Normalize the individual features of X, using their mean and standard deviation values.
# 4. Follow the usual batch gradient descent step, iterating to reduce the cost and to find an optimal theta vector.

# Dataset for this script : ex1data2.txt.
# Directory settings
curDir = os.getcwd()  # ensure that you're running this script from the parent directory i.e. ml-exercises/.
datasetPath = curDir + '/datasets/ex1data2.txt'

# Linear regression variables
X = Y = None
xmeans = []; xstddevs = []
theta = None
numFeatures = None
numSamples = None
iterations = 400
alpha = 0.01


def plotData():
  # plt.xkcd()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(X[:,1],X[:,2],Y, c='r', marker='o')
  ax.set_xlabel('\n'+'house size, in sq.m', linespacing=1.5)
  ax.set_ylabel('\n'+'no. of rooms', linespacing=1.5)
  ax.set_zlabel('\n'+'selling price, in $', linespacing=1.5)
  plt.show()


def loadData():  
  global X,Y,theta,numFeatures,numSamples
  data = pd.read_table(datasetPath, sep=',', header=None)  
  dmat = data.as_matrix()
  numFeatures = dmat.shape[1]-1  # one of them is Y
  numSamples = dmat.shape[0]
  x1 = dmat[:,0]; x2 = dmat[:,1]; y = dmat[:,2]
  # print data.describe()
  x1 = np.reshape(x1,(x1.size,1))
  x2 = np.reshape(x2,(x1.size,1))
  ones = np.ones(shape=(numSamples,1))
  X = np.concatenate((ones,x1,x2),axis=1)
  Y = np.reshape(y,(y.size,1))
  theta = np.zeros(shape=(X.shape[1],1))  # parameter vector to tune
  # histograms of given data samples.  
  # plt.xkcd()
  fig, ax = plt.subplots(1,2)
  ax[0].hist(x1,bins=10)
  ax[0].set_title('house size histogram')
  ax[0].set_xlabel('size in sq.m')
  ax[1].hist(x2,bins=10)
  ax[1].set_title('no. of rooms histogram')
  ax[1].set_xlabel('number of rooms')
  plt.show()


def scaleFeatures():
  """ Scales the features of the sample data to reduce their order of magnitude."""
  numFeatures = X.shape[1]-1
  for i in range(numFeatures):
    xmeans.append(np.mean(X[:,i+1]))
    xstddevs.append(np.nanstd(X[:,i+1]))
    X[:,i+1] -= xmeans[i]
    X[:,i+1] /= xstddevs[i]


def computeCost(X,Y,theta):
  """ Computes the cost of using a parameter vector theta."""
  H = X.dot(theta)
  diff = H-Y
  cost = sum(diff*diff)[0]
  return cost/(numSamples*2)


def runGradientDescent(X,Y,theta):
  """ Runs batch gradient descent algorithm for finding an optimal parameter vector theta."""
  # for given number of iterations, adjust theta values and compute their corresponding cost of usage
  JVals = np.zeros(shape=(iterations,1))
  thetaVals = np.zeros(shape=(iterations,theta.shape[0]))
  # print JVals.shape, thetaVals.shape
  for i in range(iterations):
    thetaVals[i] = theta.T
    H = X.dot(theta)
    sumDiff = (alpha/numSamples) * (X.T.dot(H-Y))
    theta = theta - sumDiff
    JVals[i] = computeCost(X,Y,theta)
  return (JVals, theta)


def plotCostVsIterations(JVals):
  """ Plots a graph of the costs obtained after running gradient descent vs. the number of iterations."""
  plt.figure()
  # plt.xkcd()  
  plt.plot(JVals)
  plt.xlabel('iterations')
  plt.ylabel('cost')
  plt.title('gradient descent performance')
  plt.show()


def predictPrice(input,optTheta):
  x = []
  for i in range(len(input)):
    x.append((input[i] - xmeans[i]) / xstddevs[i])
  sample = np.array([[1],[x[0]],[x[1]]])
  return sample.T.dot(optTheta)


if __name__=="__main__":
  loadData()
  plotData()
  scaleFeatures()
  JVals, optTheta = runGradientDescent(X,Y,theta)
  plotCostVsIterations(JVals)
  # print optTheta
  # Sample prediction
  price = predictPrice([1000,1],optTheta)  # price prediction for a room of 1000 sq.m and one room
  print price