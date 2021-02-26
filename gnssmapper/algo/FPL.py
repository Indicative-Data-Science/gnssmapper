"""
4 Parameter Logistic Regression
"""
from itertools import islice, accumulate

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import expit


class FourParamLogisticRegression:
    """ 4 Paramater Logistic Regression Model as described in:
    https://www.statforbiology.com/nonlinearregression/usefulequations#logistic_curve
    
    Fit through gradient descent either online (stochastically) or offline (L-BFGS-B)
    
    P(x) = d + (a-d) / ( 1 + exp(b(x-c)))
    4PL equation parameters;
    a: float, asymptotic maximum 
    b: float, gradient 
    c: float, intercept
    d: float, asymyptotic minimum 


    Parameters
    ----------
    lr : array-like, optional
        learning rate for SGD, by default [1e-6,1e-3,1e-3,1e-6]
    batch_size : int, optional
        specifies the batch size for SGD: 1 for basic SGD, larger for mini-batch., by default 100
    initial_param : list, optional
        starting parameters [a,b,c,d] for the model, by default [0.95, 1, 0, 0.05] 
    """

    def __init__(self, lr:np.array=[1e-6,1e-3,1e-3,1e-6],batch_size=100,initial_param=[0.8, 0.1, 20, 0.2]):
        self.lr = lr
        self.batch_size = batch_size
        self.param = np.array(initial_param)

        

    def _four_param_sigmoid(self,z:np.array)->np.array:
        """ Forward pass of LR function. Returns Prob(X)."""
        return self.param[3] + (self.param[0] - self.param[3]) * expit(self.param[1]*(z - self.param[2]))

    def _batch_update(self,xhat,ytrue):
        """updates model parameters using a batch of data
        Parameters
        ----------
        xhat : Training vector, 1-D only.
        ytrue : Target vector relative to X.
        """
        yhat = self._four_param_sigmoid(xhat)
        y_fac = (yhat-ytrue) / (yhat * (1 - yhat)) # nb negative log likihood for minimisation
        
        delta_a = (1/xhat.shape[0]) * np.dot(y_fac,                 (yhat - self.param[3])                  ) / (self.param[0] - self.param[3])
        delta_b = (1/xhat.shape[0]) * np.dot(y_fac, (xhat-self.param[2]) * (yhat - self.param[3]) * (self.param[0] - yhat)) / (self.param[0] - self.param[3])
        delta_c = (1/xhat.shape[0]) * np.dot(y_fac,       -self.param[1] * (yhat - self.param[3]) * (self.param[0] - yhat)) / (self.param[0] - self.param[3])
        delta_d = (1/xhat.shape[0]) * np.dot(y_fac,                                   (self.param[0] - yhat)) / (self.param[0] - self.param[3])

        self.param[0] = min(1 - 1e-3, max(0.5+1e-3, self.param[0] - (self.lr[0] * delta_a)))
        self.param[1] = max(1e-2,self.param[1] - (self.lr[1] * delta_b))
        self.param[2] = max(1e-3,self.param[2] - (self.lr[2] * delta_c))
        self.param[3] = min(0.5 - 1e-3, max(1e-3, self.param[3] - (self.lr[3] * delta_d)))
        return

    def fit_online(self, X: np.array, Y: np.array) -> np.array:
        """Returns model parameters updated by a stream of data. 

        Given a length n vecotr of traing data, returns a length n array of param estimates over time.
        
        Parameters
        ----------
        X : np.array
            Training vector, 1-D only.
        Y : np.array
            Target vector relative to X.

        Returns
        -------
        np.array
            4PL parameters


        """
        param=[self.param]
        valid= ~(np.isnan(X) | np.isnan(Y))
        xhat = X[valid]
        ytrue = Y[valid]
        for pos in range(0,xhat.shape[0],self.batch_size):
            self._batch_update(xhat[pos:pos+self.batch_size],ytrue[pos:pos+self.batch_size])
            param.append(self.param)
        idx = np.floor(np.cumsum(valid) / self.batch_size).astype('int')
        param_repeated =[param[i] for i in idx]  #this is surprisngly slow - takes as long as all batchs update. 
        param_repeated[-1]=param[-1] #takes into account the short last update
        return param_repeated

    def prob(self, X_):
        return np.where(np.isnan(X_),0,self._four_param_sigmoid(X_))

    def predict(self, X_):
        return np.where(self.prob(X_)>0.5,1,0)

    def fit_offline(self, X, Y):
        """Offline Gradient descent using entire training data
        Parameters
        ----------
        X : Training vector, 1-D only.
        y : Target vector relative to X.

        Returns
        -------
        self : object
        """
        
        valid= ~(np.isnan(X) | np.isnan(Y))
        X_ = X[valid]
        Y_ = Y[valid]
        theta_0 = self.param
        param=[self.param for _ in range(X.shape[0])]

        def neg_log_likelihood(theta, X, y):
            m = X.shape[0]
            denom_ = 1 + np.exp( - theta[1] * (X - theta[2]) )
            yhat = theta[3] + (theta[0] - theta[3])/denom_
            return -(1 / m) * np.sum(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))

        def optimize_theta(theta, X, y):
            bounds = Bounds([0.5+1e-3, 1e-3, 0, 0], [1, 100, 1000, 0.5-1e-3])
            opt_weights = minimize(neg_log_likelihood, theta, method='L-BFGS-B',bounds=bounds, args=(X, y.flatten()))
            # opt_weights = minimize(neg_log_likelihood, theta, method='Nelder-Mead', args=(X, y.flatten()))

            return opt_weights.x

        # def neg_log_likelihood_weighted(theta_, X, y):
        #     theta=np.multiply(theta_,np.array([1,100,1000,1]))
        #     m = X.shape[0]
        #     yhat = theta[3] + (theta[0] - theta[3]) *expit(theta[1] * (X - theta[2]) )
        #     return -(1 / m) * np.sum(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))

        # def optimize_theta_weighted(theta_, X, y):
        #     bounds = Bounds([0.5+1e-3, 1e-5, 0, 0], [1, 1, 1, 0.5-1e-3])
        #     theta=np.divide(theta_,np.array([1,100,1000,1]))
        #     opt_weights = minimize(neg_log_likelihood_weighted, theta, method='L-BFGS-B',bounds=bounds, args=(X, y.flatten()),options={'iprint':-1,'gtol':1e-7})
        #     # opt_weights = minimize(neg_log_likelihood, theta, method='Nelder-Mead', args=(X, y.flatten()))

        #     return np.multiply(opt_weights.x,np.array([1,100,1000,1]))


        self.param = optimize_theta(theta_0, X_, Y_)
        param[-1]=self.param
        return param

