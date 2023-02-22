import numpy as np
import pandas as pd

class Perceptron: 
    
    def __init__(self): 
        self.w = []
        self.history = []
        
    def predict(self,X):
        #returns vector of predic ted weights y_hat
        return 1*(X@self.w > 0)
    
    def score(self,X,y):
        return((np.dot(X, self.w)*y) > 0).mean()
    
    def fit(self, X, y, max_steps=1000):
        #append 1's onto X
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        #initialize random weight vector
        self.w = np.random.rand(X_.shape[1])
        
        #y into -1s and 1s for 0 and 1 respectively 
        y[y==0] = -1
        
        for m in range(max_steps):
            i = np.random.randint(X.shape[0])
            x_i = X_[i,:]
            y_i = y[i]
            w_new = self.w + 1*((y_i * np.dot(self.w, x_i)) < 0)*y_i*x_i
            
            #calculate score
            score_i = self.score(X_, y)
            self.history.append(score_i)
            
            #set weights to new weights
            self.w = w_new

    

    
