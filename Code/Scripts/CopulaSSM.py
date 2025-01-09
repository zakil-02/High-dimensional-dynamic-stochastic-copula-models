#######################################

#This was an intention of imlementation of the whole copula model proposed in the paper, but as we were not asked for this part, we did not finish it

######################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly
import sklearn
import particles
from particles import state_space_models as ssm
from particles import distributions as dists

class CopulaSSM(ssm.StateSpaceModel):
    default_params = {'n': 1, 'p': 1, 'k': 1} 

    def PX0(self):  # Distribution of X_0 
        return dists.MvNormal(loc=np.zeros(self.n), cov=np.eye(self.n))  # X_0 ~ N(0, 1)
    

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}
        # In our case the state is At.

        # Gamma : diagonal matrix
        cov_mat_eta = np.eye(self.n)
        for i in range(self.n):
            cov_mat_eta[i, i] = dists.InvGamma(a=20.0, b=0.25).rvs(size=1)
        eta = dists.Normal(0, cov_mat_eta) # eta ~ N(0, cov_mat_eta)
        # mu
        mu = np.array([dists.Normal(loc=0.4, scale=2).rvs() for _ in range(self.n)]).T.reshape(self.n, 1)
        # phi_lambda : diagonal matrix
        phi_lambda = np.eye(self.n)
        for i in range(self.n):
            phi_lambda[i, i] = dists.Normal(loc=0.985, scale=0.001).rvs(size=1)
        return dists.MvNormal(loc= mu + np.dot(phi_lambda, (xp.T - mu)), cov=cov_mat_eta) # X_t ~ N( X_{t-1}, 1)
    

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t (and X_{t-1})
        x_t = np.zeros(self.n)
        beta = np.zeros(self.k) # we take beta = 0 according to the paper
        z_t = dists.MvNormal(loc=0, cov=np.eye(self.p + self.k)).rvs(size=1).T
        epsilon_t = np.zeros(self.n)
        for i in range(self.n):
            epsilon_t[i] = dists.Normal(loc=0, scale=1).rvs(size=1)

        # Here we only use constant loading factors, we must implement a GIBBS Sampler (section in the paper)
        lambda_t = np.zeros((self.n, self.p))
        for i in range(self.n):
                lambda_t[i, :] = dists.MvNormal(loc=0.2*np.ones(self.p), cov= 2 * np.eye(self.p), scale=1).rvs(size=1)
        # volatility
        sigma_t = np.zeros(self.n)
        for i in range(self.n):
            sigma_t[i] = 1 / np.sqrt(1 + np.dot(lambda_t[i], lambda_t[i]))

        # scaling factor loadings
        for i in range(self.n):
            lambda_t[i] = lambda_t[i] / np.sqrt(1 + np.dot(lambda_t[i], lambda_t[i]))

        for i in range(self.n):
            x_t[i] = np.dot(lambda_t[i], z_t[:self.p]) + epsilon_t[i] * sigma_t[i]
        
        # u_it = P(x_it = 1) (P is gaussian cdf for now, but we must use grouped student t distribution, skewed one and compare)
        # u_t = np.zeros(self.n)
        # for i in range(self.n):
            # u_t[i] = ??
        
        # we want to return uit = P(xit | theta)
        return dists.MvNormal(loc=x_t, scale=sigma_t)