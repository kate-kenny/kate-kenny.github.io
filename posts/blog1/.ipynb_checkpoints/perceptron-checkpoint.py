import numpy as np
import pandas as pd

class Perceptron: 
    
    def __init__(self): 
        self.w = []
        self.history = []
        
    def fit(X, y):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        #initialize random weight vector
        w_t = np.random.rand(X_.shape[0])
        
        #y into -1s and 1s
        y_ = []

        for i in range(max_steps):
            x_i = X[i]
            y_i = 2*y[i] - 1
            w_t = w_t + 1*(y_i*np.dot(w_t, x_i) < 0)*y_i*x_i
        w = w_t

    def predict(X, w):
        return X@w

    #def score(X,y): 
