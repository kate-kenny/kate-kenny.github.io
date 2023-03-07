import numpy as np 

class LogisticRegression: 
        
    def __init__(self): 
        self.w = []
        self.score_history = []
        self.loss_history = []

    def sigmoid(self, z):
        #for logistic loss function 
        return 1 / (1 + np.exp(-z))
    
    def logistic_loss(self, y_hat, y): 
        #returns l(y_hat, y)
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def predict(self, X):
        #return vector of predicted y_hat
        return X@self.w
    
    def loss(self, X, y):
        #returns empirical risk of current weights on X and y
        y_hat = self.predict(X)
        return self.logistic_loss(y_hat, y).mean()
        
    def score(self, X, y):
        #returns accuracy of predictions, 1 is perfect classification 
  
        pred = self.predict(X)
        n = X.shape[0]
    
        #make both weight vectors -1's and 1's instead of 0's and 1's 
        correct = 1*(y==pred)
        
        #count # of correct predictions, divide by n, return that value
        return (1/n)*sum(correct)
    
    def gradient(self, w, X, y): 
        grad_sum = 0
        n = X.shape[0]
        
        
        #based on formula from class notes on the gradient of L(w)
        for i in range(n): 
            x_i = X[i,:]
            y_i = y[i]
            grad_sum += (self.sigmoid(np.dot(self.w, x_i)) - y_i)*x_i
            
        return (1/n)*grad_sum
                                      
    
    def fit(self, X, y, alpha = .001, max_epochs = 1000): 
        #fit method using gradient descent that updates the weight vector w
        #until the loss converges (solution to the minimum of the loss function)
        
        #append 1's onto X
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        #initialize random weight vector
        self.w = np.random.rand(X_.shape[1])
        
        done = False
        prev_loss = []
        count = 0
        
        while not done: 
            self.w = self.w - alpha*self.gradient(self.w, X_, y)
            
            new_loss = self.loss(X_, y)
            new_score = self.score(X_, y)
            
            self.loss_history.append(new_loss) 
            self.score_history.append(new_score)
            
            count += 1
            
            if count == max_epochs: 
                done = True 
                
            else: 
                prev_loss = new_loss
                
    def fit_stochastic(self, X, y, alpha=.001, batch_size = 10, max_epochs = 1000): 
        #iterates through the n points in batches (of random size) 
        #applies gradient descent to each batch 
        
        #append 1's onto X
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        n = X_.shape[0]

        #initialize random weight vector
        self.w = np.random.rand(X_.shape[1])
        
        prev_loss = []
      
        done = False 
        
        order = np.arange(n)
        count = 0 
        
        #batch process
        while not done:
            count += 1
            
            #shuffle array and select random batch size
            np.random.shuffle(order)
            batch_size = np.random.randint(2, n)
  
            done = False 
    
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                self.w = self.w - alpha*self.gradient(self.w, x_batch, y_batch)
            
            new_loss = self.loss(X_, y)
            self.loss_history.append(new_loss) 

            if (count == max_epochs): 
                done = True 
                
            else: 
                prev_loss = new_loss
                                       
        
            