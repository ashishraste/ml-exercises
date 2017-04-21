#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import PolynomialFeatures

# Steps
# 1. Load the data, create new features which are polynomials up to 6th order of the old features.
# 2. Initialize parameteric vector theta whose shape is [numFeatures x 1].
# 3. While running gradient descent, add regularization values to the cost-function and to the gradient.
# 4. Test the hypothesis' polynomial fit for different values of regularization parameter : regParam.


# Dataset for this script : ex2data2.txt.
# Directory settings
curDir = os.getcwd()  # ensure that you're running this script from the parent directory i.e. ml-exercises/.
datasetPath = curDir + '/datasets/ex2data2.txt'

X = X1 = Y = None
theta = None
numFeatures = None
numSamples = None
alpha = 0.01
regParam = 1.0
powers = None  # polynomial powers for input features


def loadData():
  """ Loads the data from the text file and initializes the matrices of input features', 
  output values and the parameter-vector. """
  global X,X1,Y,theta,numFeatures,numSamples,powers
  data = pd.read_table(datasetPath, sep=',', header=None)
  dmat = data.as_matrix()
  numFeatures = dmat.shape[1]-1  # one of them is Y
  numSamples = dmat.shape[0]
  # X = np.concatenate((np.ones((numSamples,1)),
  #   np.reshape(dmat[:,0],(numSamples,1)),
  #   np.reshape(dmat[:,1],(numSamples,1))),axis=1)\
  X = dmat[:,0:2]
  Y = np.reshape(dmat[:,2],(numSamples,1))  
  positives = np.where(Y==1)[0]
  negatives = np.where(Y==0)[0]
  ## Create new features
  powers = PolynomialFeatures(6)
  X1 = powers.fit_transform(dmat[:,0:2])  # saving the polynomial fit of input's features as X1
  theta = np.zeros((X1.shape[1],1))
  return positives,negatives


def plotData(pos,neg,optTheta=None,decisionBoundary=None):
  """ Plots the scatter-plot of the dataset, and optionally draws the decision boundary given by the computed hypothesis. """
  fig = plt.figure()
  # plt.xkcd()
  p1 = plt.scatter(X[pos,0], X[pos,1], marker='o', c='green',label='accepted')
  p2 = plt.scatter(X[neg,0], X[neg,1], marker='x', c='red',label='not accepted')
  # plt.grid()
  plt.title('Distribution of QA test scores and their acceptance result, lambda = {}'.format(regParam))
  plt.xlabel('test 1 score'); plt.ylabel('test 2 score')
  plt.legend(loc='best')
  if decisionBoundary:
    x1min = X[:,0].min() ; x1max = X[:,0].max()
    x2min = X[:,1].min() ; x2max = X[:,1].max()
    ## create xaxes, yaxes points between the min, max of the two dimensional input features
    xaxes, yaxes = np.meshgrid(np.linspace(x1min,x1max), np.linspace(x2min,x2max))
    ## calculate hypothesis for those points
    H = sigmoid(powers.fit_transform(np.c_[xaxes.ravel(),yaxes.ravel()]).dot(optTheta))  
    H = H.reshape(xaxes.shape)
    plt.contour(xaxes,yaxes,H,[0.5],linewidths=1, colors='b') 
  plt.show()


def sigmoid(Z):
  """ Returns the sigmoid function values of the given input array. """
  return (1 / (1 + np.exp(-Z)))


def computeCost(theta,X,Y,returnGradient=False):
  """ Computes the cost using a maximum likelihood function and returns the gradient for validation."""
  H = sigmoid(X.dot(theta))
  H = np.reshape(H, (-1,1))
  firstTerm = np.log(H).T.dot(Y)
  secondTerm = np.log(1-H).T.dot(1-Y)
  regTerm = regParam/(2*numSamples) * (theta[1:].T.dot(theta[1:])).sum()
  cost = -(1./numSamples) * (firstTerm + secondTerm).sum() + regTerm 
  if np.isnan(cost):
    cost = np.inf
  gradient = (1./numSamples) * X.T.dot(H-Y)  + (regParam/numSamples) * np.r_[[[0]],theta[1:].reshape(-1,1)]
  if returnGradient:
    return cost, gradient  # for validation
  return cost  # for convex optimization functions 


def runGradientDescent(X,Y,initTheta): 
  """ Runs the batch gradient using convex optimization methods, adjusting the (sigmoid) hypothesis
  to a best-possible parameter vector theta."""  
  myargs = (X,Y)
  theta = fmin(computeCost, x0=initTheta, args=myargs)  # Optimization using Nelder-Mead method.
  optTheta, optCost, _, _, _, _, _ = fmin_bfgs(computeCost, x0=initTheta, args=myargs, full_output=True)  # Optimization using BFGS.
  return optCost, np.reshape(optTheta,(optTheta.size,1))

def predict(x,optTheta):
  """ Predicts a probability for a given sample to be considered as successful or not successful. """
  return sigmoid(x.dot(optTheta))

if __name__=="__main__":
  pos,neg = loadData()
  cost = computeCost(theta, X1, Y, returnGradient=False)
  print "initial cost : ", cost
  optCost, optTheta = runGradientDescent(X1, Y, theta)
  plotData(pos, neg, optTheta, decisionBoundary=True)
  ## Sample prediction for two test scores, returns a probability of success.
  testPrediction = predict(powers.fit_transform(np.array([0.051267,0.69956])), optTheta).ravel()[0]
  