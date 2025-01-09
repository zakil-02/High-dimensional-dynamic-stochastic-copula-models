import numpy as np
import particles
import particles.state_space_models as ssm
import particles.distributions as dists
import matplotlib.pyplot as plt
from tqdm import tqdm

class CorrelatedNormals(dists.ProbDist):
    """
    Distribution class for generating correlated normal random variables.
    Uses Cholesky decomposition to generate bivariate normal with given correlation.
    """
    def __init__(self, rho):
        self.rho = rho
        # Construct correlation matrix
        self.corr_matrix = np.array([[1.0, rho], 
                                   [rho, 1.0]])
        # Compute Cholesky decomposition
        self.L = np.linalg.cholesky(self.corr_matrix)
        
    def rvs(self, size=None):
        """Generate correlated standard normal random variables"""
        if size is None:
            z = np.random.normal(0, 1, 2)
        else:
            z = np.random.normal(0, 1, (2, size))
        return np.dot(self.L, z)

class SkewedStochVolLeverage(ssm.StateSpaceModel):
    """
    Stochastic volatility model with leverage and skewed Student's t errors.
    """
    default_params = {
        'beta': np.array([0.0]),
        'gamma': 0.0,
        'phi': dists.Normal(0.985, 0.001).rvs(size=1),
        'mu': dists.Normal(0.4,2.0).rvs(size=1),
        'sigma': dists.InvGamma(20, 0.25).rvs(size=1),
        'rho': -0.7,
        'nu': 2 + dists.Gamma(2.5, 2).rvs(size=1),
        'n_lags': 5
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_history = []
        self.corr_norm = CorrelatedNormals(self.rho)
        
    def update_covariates(self, y):
        """Update the covariate matrix W with lagged values"""
        self.data_history.append(y)
        if len(self.data_history) > self.n_lags:
            self.data_history = self.data_history[-self.n_lags:]
        
        W = np.ones(self.n_lags + 1)
        for i in range(min(self.n_lags, len(self.data_history))):
            W[i+1] = self.data_history[-(i+1)]
        return W
    
    def PX0(self):
        """Initial distribution of log-volatility"""
        return dists.Normal(loc=self.mu, 
                          scale=self.sigma/np.sqrt(1 - self.phi**2))
    
    def sample_joint_innovations(self):
        """Sample correlated εy and εh innovations"""
        return self.corr_norm.rvs()
    
    def PX(self, t, xp):
        """
        Transition distribution for log-volatility using εh
        Note: eps_h is stored as self.current_eps_h from the joint sampling
        """
        if not hasattr(self, 'current_eps_h'):
            _, self.current_eps_h = self.sample_joint_innovations()
            
        # Include the innovation directly in the location
        loc = self.mu + self.phi * (xp - self.mu) + self.sigma * self.current_eps_h
        return dists.Normal(loc=loc, scale=1e-10)  # small scale for numerical stability
    
    def sample_delta(self):
        """Sample mixing variable delta from Inverse Gamma distribution"""
        return dists.InvGamma(a=self.nu/2, b=self.nu/2).rvs()
    
    def PY(self, t, xp, x):
        """
        Observation distribution with mixing variable delta and correlated eps_y
        """
        # Get covariates
        if len(self.data_history) > 0:
            W = self.update_covariates(self.data_history[-1])
        else:
            W = np.ones(len(self.beta))
        
        # Sample mixing variable δ
        delta = self.sample_delta()
        
        # Sample correlated innovations
        self.current_eps_y, self.current_eps_h = self.sample_joint_innovations()
            
        # Mean component from regression plus skewness term
        mean = np.dot(W, self.beta) + self.gamma * delta
        
        # Scale incorporating both volatility and mixing variable
        scale = np.sqrt(delta) * np.exp(x/2)
        
        # Include the innovation directly in the location
        loc = mean + scale * self.current_eps_y
        return dists.Normal(loc=loc, scale=1e-10)  # small scale for numerical stability
    
    def simulate(self, n_steps):
        """
        Simulate from the model for n_steps
        
        Returns:
        --------
        x : array
            Log-volatility process
        y : array
            Returns process
        delta : array
            Mixing variables
        """
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        delta = np.zeros(n_steps)
        eps_y = np.zeros(n_steps)
        eps_h = np.zeros(n_steps)
        
        # Initial state
        x[0] = self.PX0().rvs()
        delta[0] = self.sample_delta()
        
        # Generate subsequent states and observations
        for t in range(n_steps):
            # Sample correlated innovations
            eps_y[t], eps_h[t] = self.sample_joint_innovations()
            self.current_eps_h = eps_h[t]
            self.current_eps_y = eps_y[t]
            
            if t > 0:
                x[t] = self.PX(t, x[t-1]).rvs()
                delta[t] = self.sample_delta()
            
            # Update covariates with previous returns
            if t > 0:
                self.update_covariates(y[t-1])
                
            # Generate observation
            y[t] = self.PY(t, x[t-1] if t > 0 else None, x[t]).rvs()
            
        return x, y, delta, eps_y, eps_h



# Simulate data from the model
if __name__ == '__main__':
    for i in tqdm(range(1)):
        # Create model instance with parameters
        model = SkewedStochVolLeverage(
                beta=np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.01]),  # intercept + 5 lags
                gamma=-0.16,  # skewness
                phi= dists.Normal(0.985, 0.001).rvs(size=1),   # persistence
                mu= dists.Normal(0.4,2.0).rvs(size=1),    # mean log-volatility
                sigma= dists.InvGamma(20, 0.25).rvs(size=1), # volatility of log-volatility
                rho=0.8,   # leverage
                nu= 2 + dists.Gamma(2.5, 2).rvs(size=1),  # degrees of freedom
                n_lags=5    # number of lags
            )
        # Simulate from the model
        h, y, delta, eps_y, eps_h = model.simulate(1000)

        # Create Bootstrap particle filter
        fk_model = ssm.Bootstrap(ssm=model, data=y)
        pf = particles.SMC(fk=fk_model, N=1000)
        pf.run()
        # Plot the filtered log-volatility
        plt.figure(figsize=(10, 6))
        plt.plot(pf.X, label='Filtered', color='green')
        plt.plot(h, label='True')
        plt.title('True vs Filtered Log-Volatility')
        plt.legend()

        # Plot the delta mixing variable
        plt.figure(figsize=(10, 6))
        plt.plot(delta, label='True', color='black', linewidth=1.)
        plt.title('True')
        plt.legend()

        log_likelihood = pf.logLt
        print('Log-likelihood:', log_likelihood)

    plt.show()