import numpy as np 

class LinearRegression: 
    
    def __init__(self): 
        
        self.w = []
        self.score_history = []
    
    def predict(self, X):
        return X@self.w
    
    def score(self, X, y): 
        
        num_sum = 0
        denom_sum = 0
        y_avg = y.mean()
        y_hat = self.predict(X)
        n = X.shape[0]
        
        for i in range(n - 1): 
            num_sum += (y_hat[i] - y[i])**2
            denom_sum += (y_avg - y[i])**2
        
        
        return 1 - num_sum / denom_sum
        
    def fit_analytical(self, X, y): 
        #exact least squares regression fit method 
        
        self.w = np.linalg.inv(X.T@X)@X.T@y
        
    def fit_gradient(self, X, y, alpha = .01, max_iter = 10000): 
        #gradient descent fit method 
        
        self.w = np.random.rand(X.shape[1])

        P = X.T@X
        q = X.T@y
        
        done = False 
        count = 0
        
        while not done: 
            self.w = self.w - alpha*(P@self.w - q)
            
            new_score = self.score(X, y)
            
            self.score_history.append(new_score)
            
            count += 1
            
            if count == max_iter: 
                done = True 
                
           
        
    