import sys
import os
sys.path.append(os.path.abspath(".."))

import importlib
import tools.functions
importlib.reload(tools.functions)

import numpy as np
from tools.functions import sigmoid, threshold
from tools.metrics import binary_crossentropy


class LogisticRegression:
    
    def __init__(self):
        '''
        Initializes weights and bias
        '''
        self.weights = None
        self.bias = 0
        self.loss_history = {}
        
        
    def predict_prob(self, X):
        '''
        predicts probabilities of labelling the sample as 1 
        by performing weighted addition of the features and adding a bias
        using the current weight and bias values of the model
        '''
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)
    
    def predict(self, X):
        '''
        returns the predicted labels for the given samples
        '''
        probabilities = self.predict_prob(X)
        y = np.array([threshold(prob) for prob in probabilities])
        return y
    
    def train(self, x, y, learning_rate=0.01, epochs=10, track_loss = False):
        '''
        Trains the model and updates the weights and biases using the 
        given training data and the true labels
        '''
        n = x.shape[0]
        self.weights = np.zeros(np.shape(np.array(x))[1]) #Initializing weights
        for epoch in range(1, epochs+1):
            y_prob = self.predict_prob(x)
            d1w = np.dot(np.array(x).T, y_prob-y)/n
            d1b = np.mean(y_prob - y)
            self.weights -= learning_rate*d1w
            self.bias -= learning_rate*d1b
            
            if track_loss:
                # print("y_prob min:", y_prob.min())
                # print("y_prob max:", y_prob.max())
                # print("y_prob (first 5):", y_prob[:5])
                
                loss = binary_crossentropy(y_true = y, y_pred = y_prob)
                self.loss_history[epoch] = loss
                print(f'Epoch {epoch}, \t Loss: {loss}')
        
        return (self.weights, self.bias)
        
        
            
    
    
        
    