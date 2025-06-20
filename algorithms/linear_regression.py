import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from tools.metrics import mse

class LinearRegression:
    def __init__(self):
        """
        Initializes the Slope (b1) and Intercept (b0)
        """
        self.b1 = 0.0
        self.b0 = 0.0
        self.loss_history = {}
        
    def __repr__(self):
        return f"LinearRegression(b1={self.b1}, b0={self.b0})"
    
    def predict(self, x):
        """
        Predicts the y values using the given input x, using the current model parameters
        """
        return self.b1 * x + self.b0
    
    
    def train(self, x, y, learning_rate=1 , epochs=1, track_loss=False):
        """
        Trains the model and updates its parameters using the input x and output y by minimizing MSE loss
        """
        for epoch in range(epochs):
            
            d1b1 = -2 * np.mean(x * (y-(self.b1 * x + self.b0))) #Derivative of MSE w.r.t slope
            d1b0 = -2 * np.mean((y-(self.b1 * x + self.b0))) #Derivative of MSE loss w.r.t intercept
            
            self.b1 -= learning_rate * d1b1
            self.b0 -= learning_rate * d1b0
            
            if track_loss:
                y_pred = self.predict(x)
                loss = mse(y_true = y, y_pred = y_pred)
                self.loss_history[epoch] = loss
                print(f'Epoch {epoch+1}, \t Loss: {loss}')
        
        return (self.b0, self.b1)
            
            
            