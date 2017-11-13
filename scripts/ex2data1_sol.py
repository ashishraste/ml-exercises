#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs

# Steps
# 1. Load the data, observed samples' shape is [numSamples x numFeatures].
# 2. Initialize parameteric vector theta whose shape is [numFeatures x 1].
# 3. Run gradient descent, where instead of adjusting the theta parameter vector, make use
#    of an optimization-solver to pick alpha values, to minimize the cost function and to return an optimal theta.

# Dataset for this script : ex2data1.txt.
# Directory settings
curDir = os.getcwd()  # ensure that you're running this script from the parent directory i.e. ml-exercises/.
datasetPath = curDir + '/../datasets/ex2data1.txt'


# Logistic regression variables
X = Y = None
initTheta = None
numFeatures = None
numSamples = None
alpha = 0.01


def plotData(pos,neg,optTheta=None,decisionBoundary=False):
  """ Plots the data samples and the decision boundary given that a calculated hypothesis is available."""
  # plt.xkcd()
  fig = plt.figure()
  p1 = plt.scatter(X[pos,1], X[pos,2], marker='o', c='green')
  p2 = plt.scatter(X[neg,1], X[neg,2], marker='x', c='red')
  plt.grid()
  plt.title('Distribution of test scores and their admission results')
  plt.legend((p1,p2),('admitted','not admitted'),loc='best')

  if decisionBoundary:
    x1min = X[:,1].min() ; x1max = X[:,1].max()
    x2min = X[:,2].min() ; x2max = X[:,2].max()
    ## create xaxes, yaxes points between the min, max of the two dimensional input features
    xaxes, yaxes = np.meshgrid(np.linspace(x1min,x1max), np.linspace(x2min,x2max))
    ## calculate hypothesis for those points
    H = sigmoid(np.c_[np.ones((xaxes.ravel().shape[0],1)), xaxes.ravel(), yaxes.ravel()].dot(optTheta))
    H = H.reshape(xaxes.shape)
    plt.contour(xaxes,yaxes,H,[0.5],linewidths=1, colors='b')
  plt.show()


def loadData():
  """ Loads the data from the text file and initializes the matrices of input features',
  output values and the parameter-vector. """
  global X,Y,initTheta,numFeatures,numSamples
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
  initTheta = np.zeros((X.shape[1],1))
  positives = np.where(Y==1)[0]
  negatives = np.where(Y==0)[0]
  return positives,negatives


def sigmoid(Z):
  return (1 / (1 + np.exp(-Z)))


def computeCost(theta,X,Y,returnGradient=False):
  """ Computes the cost using a maximum likelihood function and returns the gradient for validation."""
  H = sigmoid(X.dot(theta))
  firstTerm = np.log(H).T.dot(Y)
  secondTerm = np.log(1-H).T.dot(1-Y)
  cost = -(1./numSamples) * (firstTerm + secondTerm).sum()
  # print cost
  if np.isnan(cost):
    cost = np.inf
  gradient = (1./numSamples) * X.T.dot(H-Y)
  if returnGradient:
    return cost, gradient  # for validation
  return cost  # for convex optimization functions


def runGradientDescent(X,Y,initTheta):
  """ Runs the batch gradient using convex optimization methods, adjusting the (sigmoid) hypothesis
  to a best-possible parameter vector theta."""
  cost, grad = computeCost(initTheta, X, Y, returnGradient=True)
  print "initial cost and gradient ", cost, "\n", grad
  myargs = (X,Y)
  # theta = fmin(computeCost, x0=initTheta, args=myargs)  # Optimization using Nelder-Mead method.
  optTheta, optCost, _, _, _, _, _ = fmin_bfgs(computeCost, x0=initTheta, args=myargs, full_output=True)  # Optimization using BFGS.
  return optCost, np.reshape(optTheta,(optTheta.size,1))


if __name__=="__main__":
  pos,neg = loadData()
  plotData(pos,neg)
  J, optTheta = runGradientDescent(X,Y,initTheta)
  ## Sample prediction
  prob = sigmoid(np.dot(np.array([1,45,85]),optTheta))
  print prob
  ## Data plot with decision boundary
  plotData(pos,neg,optTheta,True)
