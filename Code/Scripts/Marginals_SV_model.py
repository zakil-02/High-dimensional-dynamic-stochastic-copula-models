import numpy as np
import particles
import particles.state_space_models as ssm
import particles.distributions as dists
import matplotlib.pyplot as plt
from tqdm import tqdm

class SkewedStochVolLeverage(ssm.StateSpaceModel):
    """
    Stochastic volatility model with leverage and skewed Student's t errors.
    
    Parameters:
    -----------
    beta : array-like
        Regression parameters for exogenous covariates
    gamma : float
        Skewness parameter
    phi : float
        Persistence parameter for log-volatility
    mu : float
        Mean of log-volatility
    sigma : float
        Volatility of log-volatility
    rho : float
        Leverage parameter (correlation between return and volatility innovations)
    nu : float
        Degrees of freedom for Student's t distribution
    n_lags : int
        Number of lags to include as covariates
    """
    default_params = {
        'beta': np.array([0.0]),  # Initialize with just intercept
        'gamma': 0.0,
        'phi': 0.98,
        'mu': 0.0,
        'sigma': 0.15,
        'rho': -0.7,
        'nu': 7.0,
        'n_lags': 5
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_history = []
        
    def update_covariates(self, y):
        """Update the covariate matrix W with lagged values"""
        self.data_history.append(y)
        if len(self.data_history) > self.n_lags:
            self.data_history = self.data_history[-self.n_lags:]
        
        # Construct W matrix: [1, y_{t-1}, ..., y_{t-n_lags}]
        W = np.ones(self.n_lags + 1)
        for i in range(min(self.n_lags, len(self.data_history))):
            W[i+1] = self.data_history[-(i+1)]
        return W
    
    def PX0(self):
        """Initial distribution of log-volatility"""
        return dists.Normal(loc=self.mu, 
                          scale=self.sigma/np.sqrt(1 - self.phi**2))
    
    def PX(self, t, xp):
        """Transition distribution for log-volatility"""
        return dists.Normal(loc=self.mu + self.phi * (xp - self.mu),
                          scale=self.sigma)
    
    def sample_delta(self):
        """Sample mixing variable δ from Inverse Gamma distribution"""
        return dists.InvGamma(a=self.nu/2, b=self.nu/2).rvs()
    
    def PY(self, t, xp, x):
        """
        Observation distribution: skewed Student's t with time-varying volatility
        and leverage effect
        """
        # Get covariates
        if len(self.data_history) > 0:
            W = self.update_covariates(self.data_history[-1])
        else:
            W = np.ones(len(self.beta))
            
        # sample delta 
        delta = self.sample_delta()

        # Mean component from regression
        mean = np.dot(W, self.beta) + self.gamma * delta

        # Scale incorporating volatility
        scale = np.exp(x/2) * np.sqrt(delta)
        
        # Create joint normal distribution for return and volatility innovations
        # to incorporate leverage effect
        return dists.Normal(loc=mean, scale=scale)
    
    def simulate(self, n_steps):
        """
        Simulate from the model for n_steps
        
        Returns:
        --------
        x : array
            Log-volatility process
        y : array
            Returns process
        """
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        
        # Initial state
        x[0] = self.PX0().rvs()
        
        # Generate subsequent states and observations
        for t in range(n_steps):
            if t > 0:
                x[t] = self.PX(t, x[t-1]).rvs()
            
            # Update covariates with previous returns
            if t > 0:
                self.update_covariates(y[t-1])
                
            # Generate observation
            y[t] = self.PY(t, x[t-1] if t > 0 else None, x[t]).rvs()
            
        return x, y

class SkewedStudentT(dists.ProbDist):
    """
    Skewed Student's t distribution
    
    Parameters:
    -----------
    loc : float
        Location parameter
    scale : float
        Scale parameter
    df : float
        Degrees of freedom
    gamma : float
        Skewness parameter
    """
    def __init__(self, loc=0., scale=1., df=7., gamma=0.):
        self.loc = loc
        self.scale = scale
        self.df = df
        self.gamma = gamma
        
    def rvs(self, size=None):
        """Generate random variates"""
        z = dists.StudentT(df=self.df).rvs(size=size)
        delta = self.gamma * np.abs(z)
        return self.loc + self.scale * (z + delta)
    
    def logpdf(self, x):
        """Compute log-density"""
        z = (x - self.loc) / self.scale
        return (dists.StudentT(df=self.df).logpdf(z) - 
                np.log(self.scale) +
                np.log(1 + self.gamma * np.sign(z)))



if __name__ == '__main__':
    for i in tqdm(range(5)):
        # Create model instance with parameters

        model = SkewedStochVolLeverage(
            beta=np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.01]),  # intercept + 5 lags
            gamma=-0.14,  
            phi= dists.Normal(0.985, 0.001).rvs(size=1),   # persistence
            mu= dists.Normal(0.4,2.0).rvs(size=1),    # mean log-volatility
            sigma= dists.InvGamma(20, 0.25).rvs(size=1), # volatility of log-volatility
            rho=-0.7,   # leverage
            nu= 2 + dists.Gamma(2.5, 2).rvs(size=1),  # degrees of freedom
            n_lags=5    # number of lags
        )
        # Simulate from the model
        h, y = model.simulate(1000)

        # Create Bootstrap particle filter
        fk_model = ssm.Bootstrap(ssm=model, data=y)
        pf = particles.SMC(fk=fk_model, N=1000)
        pf.run()
        # Plot the filtered log-volatility
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plt.plot(pf.X, label='Filtered', color='yellow', alpha=0.7, linewidth=0.4)
        plt.plot(h, label='True', color='blue', linestyle='--', linewidth=0.4)
        plt.title('True vs Filtered Log-Volatility')
        plt.legend()
    plt.show()