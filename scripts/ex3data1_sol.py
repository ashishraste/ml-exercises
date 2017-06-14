#!/usr/bin/python

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


# Steps:
# 1. Import the sample dataset, initialize X and Y matrices, append ones to input samples X.
# 2. Initialize a theta-matrix, which will have the optimized theta parameter vectors, one for each of the classifier 
#    targeting a single digit from 0-9.
# 3. Run gradient descent using sigmoid hypothesis, using convex optimization techniques to find the optimum theta 
#    parameter vectors.
# 4. Test the trained classifiers.
# 5. Shapes of matrices for reference:
#    X = [5000 x 400] -> [5000 x 401] (additional column ones for X0).
#    Y = [5000 x 1]             # has class labels whose values range from 1-10 for images having numbers 0-9 respectively.
#    theta vector = [401 x 1]   # a single parameter vector for a single class label.
#    theta matrix = [10 x 401]  # a matrix containing all the theta parameter vectors, each row for each of the 10 class labels. 


# globals, classifier parameters
X = Y = None
numSamples = numFeatures = None
regParam = 0.1

# dataset path
curDir = os.getcwd()
datasetPath = curDir + '/datasets/ex3data1.mat'


def sigmoid(Z):
  ''' 
  Returns the sigmoid function values of the given input array. 
  '''
  return (1 / (1 + np.exp(-Z)))


def computeCost(theta,X,Y):
  '''
  Computes the cost using a maximum likelihood function and returns the gradient for validation.
  Args:
    theta (numpy ndarray) : The parameter vector corresponding to a single classifier (for one given class-label).
  '''
  H = sigmoid(X.dot(theta))
  # H = H.reshape(-1,1)
  firstTerm = np.log(H).T.dot(Y)
  secondTerm = np.log(1-H).T.dot(1-Y)
  regTerm = regParam/(2*numSamples) * (theta[1:].T.dot(theta[1:])).sum()
  cost = -(1./numSamples) * (firstTerm + secondTerm).sum() + regTerm 
  if np.isnan(cost):
    cost = np.inf
  return cost


def computeGradient(theta,X,Y):
  '''
  Computes the gradient of the cost function.
  Take note of reshaping theta to its original shape and flattening the gradient values.
  Returns:
    gradient (numpy ndarray) : The computed gradient. A 1-D array having size equal to the number of features. 
  '''
  H = sigmoid(X.dot(theta.reshape(-1,1)))
  gradient = (1./numSamples) * (X.T.dot(H-Y)) + (regParam/numSamples) * np.r_[[[0]],theta[1:].reshape(-1,1)]
  return gradient.flatten()


def trainClassifier(X,Y,numClasses):
  '''
  Trains the multi-class classifier. Runs BFGS convex-optimization making use of computeGradient and computeCost functions.
  Args:
    X (numpy ndarray) : The dataset's input samples along with their pixel-value features.
    Y (numpy ndarray) : The dataset's output values (labels).
    numClasses (int)  : Number of classes in the dataset. We are having 10 classes in our dataset.

  Returns: 
    thetaMatrix (numpy ndarray) : A 2-D array having the trained parameter vector for each of the 10 classes.
  '''
  initTheta = np.zeros((X.shape[1],1))
  thetaMatrix = np.zeros((numClasses,X.shape[1]))
  for classLabel in np.arange(1,numClasses+1):
    optTheta = minimize(computeCost, initTheta, args=(X,(Y==classLabel)*1), method=None, jac=computeGradient, options={'maxiter':75})
    thetaMatrix[classLabel-1] = optTheta.x  # contains the optimized parameter vector.
  return thetaMatrix


def predictAccuracy(X,theta):
  probs = sigmoid(X.dot(theta.T))
  return (np.argmax(probs, axis=1)+1)  # finds the column (class-label) with the maximum probability, for each of the 5000 samples.


def loadData():
  '''
  Loads the dataset, initializes the input-samples' and output-labels' matrices.
  '''
  global X,Y,numSamples,numFeatures,theta
  mat = loadmat(datasetPath)
  X = mat['X']; Y = mat['y']
  numSamples = X.shape[0]
  X = np.c_[np.ones((numSamples,1)),X]
  

def visualizeSamples():
  '''
  Visualize few random samples out of the dataset.
  '''
  random = np.random.choice(numSamples,5)
  plt.imshow(X[random,1:].reshape(-1,20).T)
  plt.gray(); plt.axis('off')
  plt.show()


if __name__=="__main__":
  loadData()
  visualizeSamples()  # plot some random samples
  thetaMatrix = trainClassifier(X, Y, 10)
  prediction = predictAccuracy(X, thetaMatrix) 
  print 'training accuracy = {}%'.format(np.mean(prediction == Y.ravel())*100)


