"""
4 Parameter Logistic Regression
"""
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import expit
from itertools import islice,accumulate
from math import floor

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
        return self.d + (self.a - self.d) * expit(self.b*(z - self.c))
        
    def fit_online_iter(self, X, Y):
        """Fits the model using a data stream
        Parameters
        ----------
        X : Training vector, 1-D only.
        y : Target vector relative to X.

        Yields
        -------
        params: [4] list of 4PL paramters 
        """
        xhat=[]
        ytrue=[]
        xit=iter(X)
        yit=iter(Y)

        # grabs an initial data point but doesn't yield anything. this is so a final yield can happen with a short batch update if needed. 
        try:
            x_prior=next(xit)
            y_prior=next(yit)
        except StopIteration:
            return

        
        for x,y in zip(xit,yit):
            if ~(np.isnan(x_prior) | np.isnan(y_prior)):
                xhat.append(x_prior)
                ytrue.append(y_prior)
            if len(xhat)==self.batch_size:
                self.batch_update(np.array(xhat),np.array(ytrue))
                xhat=[]
                ytrue=[]
            x_prior=x
            y_prior=y
            yield [self.a,self.b,self.c,self.d]

        
        if ~(np.isnan(x_prior) | np.isnan(y_prior)):
            xhat.append(x_prior)
            ytrue.append(y_prior)
        if len(xhat)>0:
            self.batch_update(np.array(xhat),np.array(ytrue))
        yield [self.a,self.b,self.c,self.d]

    def batch_update(self,xhat,ytrue):
        """updates model parameters using a batch of data
        Parameters
        ----------
        xhat : Training vector, 1-D only.
        ytrue : Target vector relative to X.
        """
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
        return

    def fit_online(self, X, Y):
        """Fit the model using a data stream
        Parameters
        ----------
        X : Training vector, 1-D only.
        Y : Target vector relative to X.

        Returns
        -------
        params: [n,4] array of 4PL paramters where data points
        """
        param=[[self.a,self.b,self.c,self.d]]
        valid= ~(np.isnan(X) | np.isnan(Y))
        xhat = X[valid]
        ytrue = Y[valid]
        for pos in range(0,xhat.shape[0],self.batch_size):
            self.batch_update(xhat[pos:pos+self.batch_size],ytrue[pos:pos+self.batch_size])
            param.append([self.a,self.b,self.c,self.d])
        idx = np.floor(np.cumsum(valid)/self.batch_size).astype('int')
        param_repeated =[param[i] for i in idx]  #this is surprisngly slow - takes as long as all batchs update. 
        param_repeated[-1]=param[-1] #takes into account the short last update
        return param_repeated

    def prob(self, X_):
        return np.where(np.isnan(X_),0,self.four_param_sigmoid(X_))

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
        theta_0 = np.array([self.a, self.b, self.c, self.d])
        param=[[self.a,self.b,self.c,self.d] for _ in range(X.shape[0])]

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

        def neg_log_likelihood_weighted(theta_, X, y):
            theta=np.multiply(theta_,np.array([1,100,1000,1]))
            m = X.shape[0]
            yhat = theta[3] + (theta[0] - theta[3]) *expit(theta[1] * (X - theta[2]) )
            return -(1 / m) * np.sum(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))

        def optimize_theta_weighted(theta_, X, y):
            bounds = Bounds([0.5+1e-3, 1e-3, 0, 0], [1, 1, 1, 0.5-1e-3])
            theta=np.divide(theta_,np.array([1,100,1000,1]))
            opt_weights = minimize(neg_log_likelihood_weighted, theta, method='L-BFGS-B',bounds=bounds, args=(X, y.flatten()))
            # opt_weights = minimize(neg_log_likelihood, theta, method='Nelder-Mead', args=(X, y.flatten()))

            return np.multiply(opt_weights.x,np.array([1,100,1000,1]))


        self.a, self.b, self.c, self.d = optimize_theta_weighted(theta_0, X_, Y_)
        param[-1]=[self.a,self.b,self.c,self.d] 
        return param

