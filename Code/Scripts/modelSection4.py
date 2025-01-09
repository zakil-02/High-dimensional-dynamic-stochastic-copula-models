import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly
import sklearn
import particles
from particles import state_space_models as ssm
from particles import distributions as dists

class Sec4SSM(ssm.StateSpaceModel):
    default_params = {'n': 1, 'p': 3, 'k': 1}

    def __init__(self, n=1, p=3, k=1):
        self.n = n
        self.p = p
        self.k = k
        W_t = np.zeros((n, p + 1))
        W_t[:, 0] = 1.0
        self.W_t = W_t


    def PX0(self):  # Distribution of X_0 
        return dists.MvNormal(loc=np.zeros(self.n), cov=np.eye(self.n))  # X_0 ~ N(0, 1)
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}
        epsilon_h_t = np.array(dists.Normal(loc=0, scale=1).rvs(size=self.n)).reshape(self.n, 1)
        sigma_h = np.array(dists.InvGamma(a=20.0, b=0.25).rvs(size=self.n)).reshape(self.n, 1)
        mu = np.array(dists.Normal(loc=0.4, scale=2).rvs(size=self.n)).reshape(self.n, 1)
        phi_lambda = np.eye(self.n)
        for i in range(self.n):
            phi_lambda[i, i] = dists.Normal(loc=0.985, scale=0.001).rvs(size=1)
        
        # print(np.diag((sigma_h * epsilon_h_t).flatten())) # not positive definite !!!! 

        return dists.MvNormal(loc=mu + np.dot(phi_lambda, (xp.T - mu)), cov = np.diag(sigma_h.flatten()**2))
    
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t (and X_{t-1})
        # x is the state at time t (X_t) and xp is the state at time t-1 (X_{t-1}) 
        # y = Wb + gamma * delta + scale * epsilon 
        epsilon_y_t = np.array(dists.Normal(loc=0, scale=1).rvs(size=self.n)).reshape(self.n, 1)
        nu_t = np.array(dists.Gamma(a=2.5, b=2).rvs(size=self.n)).reshape(self.n, 1) + 2 #shifted gamma
        delta_t = np.array([dists.InvGamma(a=nu_it/2, b=nu_it/2).rvs(size=1) for nu_it in nu_t]).reshape(self.n, 1)
        beta_t  = dists.MvNormal(loc=np.zeros(self.p + 1), cov=0.5*np.eye(self.p + 1), scale=1).rvs(size=1).reshape(self.p + 1, 1) # regression coefficients
        gamma_y = dists.MvNormal(loc=np.zeros(self.n), cov=np.eye(self.n), scale=1).rvs(size=1).reshape(self.n, 1) # skewness coefficients
        # print(beta_t.shape)
        # print(self.W_t.shape)
        print("prod", np.dot(self.W_t,beta_t).shape)
        y_t = np.dot(self.W_t,beta_t) + gamma_y * delta_t + np.sqrt(delta_t) * np.exp(x/2).T * epsilon_y_t

        # UPDATE THE EXOGENOUS VARIABLES
        # we must forget the earliest value and add the new value of y
        W = self.W_t
        W = np.delete(W, 1, axis=1)

        W = np.append(W, y_t, axis=1)
        self.W_t = W



        return dists.MvNormal(loc=y_t)

# Instantiate and simulate the model
model = Sec4SSM(n=200)
states, observations = model.simulate(T=100)
# print(states)
# print(observations)

