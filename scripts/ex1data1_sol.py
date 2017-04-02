
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
from os.path import expanduser

# Steps
# 1. Start with a zero parameter vector (theta) having 2 parameters -> [2x1] matrix.
# 2. Supply it to the computeCost function, along with known input matrix X [mx2] and
#    known output values Y [mx1], to see how theta vector performs.
# 3. Compute Hypothesis function H from theta and known input X. It is represented as a [mx1] matrix.
# 4. Run batch gradient descent, with a given number of iterations, where the parametric values of theta are 
#    simultaneously updated to reduce the cost. 

# Dataset for this script : ex1data1.txt.
# Directory settings
curDir = os.getcwd()  # ensure that you're running this script from the parent directory i.e. ml-exercises/.
datasetPath = curDir + '/datasets/ex1data1.txt'

### Default parameters for linear regression
alpha = 0.01
iterations = 1500


def computeCost(X,Y,theta):
  """ Computes cost of a hypothesis function H."""
  H = X.dot(theta)
  numSamples = len(Y)  
  diff = H-Y
  cost = sum(diff**2)[0]  # computes sum squared value of the difference H-Y.
  return cost / (2*numSamples)


def runGradientDescent(X,Y,theta):
  """ Runs a batch gradient descent algorithm, trying to find the optimal parameter vector theta."""
  numSamples = len(Y)
  JVals = np.zeros(iterations)
  thetaVals = np.zeros(shape=(iterations,2))
  for i in range(iterations):
    thetaVals[i] = theta.T
    H = X.dot(theta)
    sumDelta = (alpha/numSamples) * (X.T.dot(H-Y))  # returns [1x2] matrix
    theta = theta - sumDelta
    JVals[i] = computeCost(X, Y, theta)
    # print JVals[i]
  return (JVals, theta)
  

def plotDataset(X,Y):
  """ Plots a scatter-plot between the known input and output values of the samples."""
  # plt.xkcd()
  fig = plt.figure()
  plt.scatter(X,Y)
  plt.xlabel('population of city in 10000s');
  plt.ylabel('profit in $10000s')
  plt.show()


def plotCostVsIterations(JVals):
  """ Plots a graph of the costs obtained after running gradient descent vs. the number of iterations."""
  # plt.xkcd()
  fig = plt.figure()
  plt.plot(JVals)
  plt.xlabel('iterations')
  plt.ylabel('cost')
  plt.show()


if __name__=="__main__":
  ### Load data
  data = pd.read_table(datasetPath, sep=',', header=None)
  data.describe()
  dmat = data.as_matrix()
  x = dmat[:,0]; Y = dmat[:,1]
  x = np.reshape(x,(len(x),1))
  Y = np.reshape(Y,(len(Y),1))
  plotDataset(x,Y)
  
  ### Run linear regression using batch gradient descent for finding the best theta vector.
  m = len(Y) # number of samples
  ones = np.ones(shape=(m,1))
  zeros = np.zeros(shape=(m,1))
  X = np.concatenate((ones,x),axis=1)
  theta = np.zeros(shape=(2,1))
  JVals, optTheta = runGradientDescent(X,Y,theta)
  plotCostVsIterations(JVals)

  ### Predict the profit for a given population of a city.
  print("sample profit prediction for a population of 40K : ${0:.2f}".format(np.array([[1,4]]).dot(optTheta)[0][0]*10000)) # population and profit value returned are in 10000s.