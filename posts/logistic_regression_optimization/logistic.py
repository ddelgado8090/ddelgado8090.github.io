import torch

class LinearModel:
    """This is code I got from the warm up we did for Linear Models
    """
    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # computing the vector of scores s
        scores = (X@self.w)
        return scores

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        y_hat = torch.where(scores >= 0, torch.tensor(1.0), torch.tensor(0.0))
        return y_hat
    

class LogisticRegression(LinearModel):
    """LogisticRegression inherits form LinearModel"""
    def __init__ (self):
        """
       Initializes weight vector w to none
         """
        #initialize w
        self.w = None
    def loss(self, X, y):
        """
        Computes the empirical risk L(w) using the logistic loss formula.
        Looks at the first 25 data points from scores before doing the logistic loss formula.

        Inputs:
        X, torch.Tensor: the feature matrix
        y, torch.Tensor: target labels

        Returns: torch.Tensor: The logistic loss value
        """
        if self.w is None:
            #gives a random value to w
            #sigmas turning to one, which then log(1-1) is zero so gives infinity--> divide by X.size
            self.w = torch.rand((X.size()[1]))/X.size()[1]
        s = X @ self.w
        sigma_s = 1 / (1 + torch.exp(-s))
        logistic_loss = torch.mean(-y * torch.log(sigma_s) - (1 - y) * torch.log(1 - sigma_s))
        return logistic_loss
    
    def grad(self, X, y):
        """
        Computes the gradient of the empirical risk L(w) using the gradient formula
        
        Inputs:
        X, torch.Tensor: the feature matrix
        y, torch.Tensor: target labels

        Returns: torch.Tensor: The gradient of the logistic loss function
        """
        if self.w is None:
            #sets the value for the weight
            self.w = torch.rand((X.size()[1]))

        s = X @ self.w
        sigma_s = 1 / (1 + torch.exp(-s))
        logistic_gradient = torch.mean((sigma_s - y)[:, None] * X, dim = 0)
        gradient_matrix = logistic_gradient[:, None]
        return gradient_matrix
    
class GradientDescentOptimizer:
    def __init__(self, model):
        """
        Initializes the gradient decent optimizer model.
        Initializes the old_w that will hold the previous value of w.

        Inputs:
            model: the model to be used in the optimizer class
        """
        self.model = model
        self.old_w = None

    def step(self, X, y, alpha, beta):
        """
        This does the gradient descent optimizer step with momentum where the old_w and current_w help implement the momentum.

        Inputs:
        X, torch.Tensor: the feature matrix
        y, torch.Tensor: target labels
        alpha (float): learning rate parameter
        beta(float): learning rate parameter - momentum rate parameter
        """
        if self.old_w is None:
            self.old_w = self.model.w.clone() #setting old_w to be the cloned version of w
        current_w_temp = self.model.w.clone() #getting the current weights
        self.model.w = self.model.w - self.model.grad(X, y).flatten() * alpha + beta * (self.model.w - self.old_w)
        self.old_w = current_w_temp
        

