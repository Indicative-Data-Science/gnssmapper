"""
4 Parameter Logistic Regression
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from itertools import islice


class FourParamLogisticRegression:
    """ 4 Paramater Logistic Regression Model as described in:
        https://www.statforbiology.com/nonlinearregression/usefulequations#logistic_curve
    
    Fit through gradient descent either online (stochastically) or offline (L-BFGS-B)
    
    Parameters
    ----------

    lr : float,  default = 1e-5, learning rate for SGD
    batch_size: int, specifies the batch size for SGD: 1 for basic SGD, larger for mini-batch.

    Attributes
    ----------
    4PL equation parameters;
        a: float, asymptotic maximum 
        b: float, gradient 
        c: float, intercept
        d: float, asymyptotic minimum 

    """
    def __init__(self, lr=np.array([1e-6,1e-3,1e-3,1e-6]),batch_size=100):
        self.lr = lr
        self.batch_size = batch_size

        # Initial parameters
        self.a, self.b, self.c, self.d = 0.95, 1, 0, 0.05

    def four_param_sigmoid(self,z):
        """ Forward pass of LR function
        Parameters
        ----------
        z: [n,] np.array 

        Returns
        -------
        probability: [n,] np.array
         """
        denom = 1 + np.exp( - self.b*(z - self.c))
        result = self.d + (self.a - self.d) / denom
        return result
        

    def fit_online(self, X, y):
        """Fit the model using a data stream
        Parameters
        ----------
        X : Training vector, 1-D only.
        y : Target vector relative to X.

        Returns
        -------
        self : object
        """
        param=[]
        for pos in range(0,X.shape[0],self.batch_size):
            xhat = X[pos:pos+self.batch_size]
            ytrue = y[pos:pos+self.batch_size]
            missing=np.isnan(xhat)
            xhat = xhat[~missing]
            ytrue = ytrue[~missing]
            if xhat.shape[0]>0:
                yhat = self.four_param_sigmoid(xhat)
                y_fac = (yhat-ytrue) / (yhat * (1 - yhat)) # nb negative log likihood for minimisation
                
                delta_a = (1/xhat.shape[0]) * np.dot(y_fac,                 (yhat - self.d)                  ) / (self.a - self.d)
                delta_b = (1/xhat.shape[0]) * np.dot(y_fac, (xhat-self.c) * (yhat - self.d) * (self.a - yhat)) / (self.a - self.d)
                delta_c = (1/xhat.shape[0]) * np.dot(y_fac,       -self.b * (yhat - self.d) * (self.a - yhat)) / (self.a - self.d)
                delta_d = (1/xhat.shape[0]) * np.dot(y_fac,                                   (self.a - yhat)) / (self.a - self.d)

                self.a = min(1 - 1e-3, max(0.5+1e-3, self.a - (self.lr[0] * delta_a)))
                self.b = max(1e-2,self.b - (self.lr[1] * delta_b))
                self.c = max(1e-3,self.c - (self.lr[2] * delta_c))
                self.d = min(0.5 - 1e-3, max(1e-3, self.d - (self.lr[3] * delta_d)))
            param.append([self.a,self.b,self.c,self.d])

        return np.array(param)

    def prob(self, X_):
        return np.where(np.isnan(X_),0,self.four_param_sigmoid(X_))

    def predict(self, X_):
        return np.where(self.prob(X_)>0.5,1,0)

    def fit_offline(self, X, y):
        """Offline Gradient descent using entire training data
        Parameters
        ----------
        X : Training vector, 1-D only.
        y : Target vector relative to X.

        Returns
        -------
        self : object
        """
        
        missing=np.isnan(X)
        X_ = X[~missing]
        y_ = y[~missing]
        theta_0 = np.array([self.a, self.b, self.c, self.d])

        def neg_log_likelihood(theta, X, y):
            m = X.shape[0]
            denom_ = 1 + np.exp( - theta[1] * (X - theta[2]) )
            yhat = theta[3] + (theta[0] - theta[3])/denom_
            return -(1 / m) * np.sum(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))

        def optimize_theta(theta, X, y):
            bounds = Bounds([0.5+1e-3, 1e-3, 0, 0], [1, 100, 1000, 0.5-1e-3])
            opt_weights = minimize(neg_log_likelihood, theta, method='L-BFGS-B',bounds=bounds, args=(X, y.flatten()))

            return opt_weights.x

        self.a, self.b, self.c, self.d = optimize_theta(theta_0, X_, y_)

        return self

