import numpy as np
from scipy import stats
from typing import Tuple, List, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import particles

class FactorLoadingParticleGibbs:
    def __init__(self, 
                 n: int,  # Number of series
                 p: int,  # Number of factors
                 T: int,  # Time series length
                 M: int = 100):  # Number of particles
        self.n = n
        self.p = p
        self.T = T
        self.M = M
        self.np = n * p
        
    def initialize_state(self, theta: dict) -> np.ndarray:
        """
        Initialize state vector following the paper's specification
        Lambdaᵢ,₁ ~ N(mu+i, 100*Σᵢᵢ)
        """
        mu = theta['mu']  # np x 1
        sigma = theta['sigma']  # np x 1 (diagonal elements)
        
        # Initialize with inflated variance as per paper
        return np.random.normal(
            mu.reshape(-1),
            np.sqrt(100 * sigma.reshape(-1))
        )
    
    def transition_density(self,
                         lambda_t: np.ndarray,
                         lambda_prev: np.ndarray,
                         theta: dict) -> float:
        """
        Transition density p(λₜ|λₜ₋₁,θ) with diagonal Φλ and Σ
        """
        mu = theta['mu']  # np x 1
        phi = theta['phi']  # np x 1 (diagonal elements of Φλ)
        sigma = theta['sigma']  # np x 1 (diagonal elements of Σ)
        
        mean = phi * lambda_prev + (1 - phi) * mu
        return np.sum(stats.norm.logpdf(lambda_t, mean, np.sqrt(sigma)))
    
    def observation_density(self,
                          ut: np.ndarray,
                          Xt: np.ndarray,
                          zt: np.ndarray,
                          zeta_t: float,
                          lambda_t: np.ndarray,
                          theta: dict) -> float:
        """
        Observation density p(uₜ|z_t,X_t,lambda_t,zeta_t,θ)
        """
        # Reshape lambda_t to n x p matrix
        lambda_mat = lambda_t.reshape(self.n, self.p)
        
        # Calculate scaled loadings as per equation (6)
        lambda_norm = np.sum(lambda_mat**2, axis=1, keepdims=True)
        lambda_scaled = lambda_mat / np.sqrt(1 + lambda_norm)
        
        # Calculate likelihood based on model specification
        # This is a simplified version - adjust based on your specific copula choice
        return np.sum(stats.norm.logpdf(ut, np.dot(lambda_scaled, zt), np.sqrt(zeta_t)))
    
    def propose_particles(self,
                         lambda_prev: np.ndarray,
                         theta: dict) -> np.ndarray:
        """
        Proposal distribution q
        """
        mu = theta['mu']
        phi = theta['phi']
        sigma = theta['sigma']
        
        # Use transition density as proposal
        mean = phi * lambda_prev + (1 - phi) * mu
        return np.random.normal(mean, np.sqrt(sigma))
    
    def conditional_resampling(self,
                             particles: np.ndarray,
                             weights: np.ndarray,
                             reference_particle: np.ndarray) -> np.ndarray:
        """
        Conditional multinomial resampling ensuring reference trajectory survives
        """
        M = self.M
        resampled = np.zeros((M, self.np))
        
        # Keep reference particle
        resampled[0] = reference_particle
        
        # Resample M-1 particles
        indices = np.random.choice(M, size=M-1, p=weights)
        resampled[1:] = particles[indices]
        
        return resampled
    
    def backward_sampling(self,
                         particles_history: List[np.ndarray],
                         weights_history: List[np.ndarray],
                         theta: dict) -> np.ndarray:
        """
        Backward sampling to draw trajectory
        """
        trajectory = np.zeros((self.T, self.np))
        
        # Sample last state
        idx = np.random.choice(self.M, p=weights_history[-1])
        trajectory[-1] = particles_history[-1][idx]
        
        # Backward pass
        for t in range(self.T-2, -1, -1):
            back_weights = np.zeros(self.M)
            
            # Calculate backward weights
            for m in range(self.M):
                back_weights[m] = weights_history[t][m] + \
                                self.transition_density(trajectory[t+1],
                                                      particles_history[t][m],
                                                      theta)
                                
            # Normalize weights
            back_weights = np.exp(back_weights - np.max(back_weights))
            back_weights /= np.sum(back_weights)
            
            # Sample state
            idx = np.random.choice(self.M, p=back_weights)
            trajectory[t] = particles_history[t][idx]
            
        return trajectory
    
    def sample_path(self,
                   ut: np.ndarray,  # T x n matrix of observations
                   Xt: np.ndarray,  # T x n x k matrix of covariates
                   zt: np.ndarray,  # T x p matrix of factors
                   zeta_t: np.ndarray,  # T vector of mixing variables
                   theta: dict,
                   reference_path: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run particle Gibbs sampler for one complete path
        """
        particles_history = []
        weights_history = []
        
        # Initialize particles
        particles = np.array([self.initialize_state(theta) for _ in range(self.M)])
        
        if reference_path is not None:
            particles[0] = reference_path[0]
            
        for t in range(self.T):
            # Store previous particles
            prev_particles = particles.copy()
            
            # Propose new particles (except reference)
            if reference_path is not None:
                particles[0] = reference_path[t]
            for m in range(1 if reference_path is not None else 0, self.M):
                particles[m] = self.propose_particles(prev_particles[m], theta)
            
            # Calculate weights
            log_weights = np.zeros(self.M)
            for m in range(self.M):
                # Numerator: p(uₜ|zₜ,Xₜ,λₜ,ζₜ,θ)p(λₜ|λₜ₋₁,θ)
                log_weights[m] = self.observation_density(ut[t], Xt[t], zt[t], zeta_t[t], 
                                                        particles[m], theta)
                log_weights[m] += self.transition_density(particles[m], prev_particles[m], theta)
                
                # Denominator: proposal density (if different from transition)
                # Not needed here as we use transition density as proposal
            
            # Normalize weights
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)
            
            # Conditional resampling
            if reference_path is not None:
                particles = self.conditional_resampling(particles, weights, reference_path[t])
            
            particles_history.append(particles.copy())
            weights_history.append(weights.copy())
            
        # Backward sampling
        return self.backward_sampling(particles_history, weights_history, theta)


# Function to run the complete Particle Gibbs sampler
def run_factor_loading_sampler(ut: np.ndarray,
                             Xt: np.ndarray,
                             zt: np.ndarray,
                             zeta_t: np.ndarray,
                             n: int,
                             p: int,
                             n_iterations: int = 1000) -> np.ndarray:
    """
    Run complete Particle Gibbs sampling for factor loadings
    Returns: Array of shape (n_iterations, T, n*p) containing sampled paths
    """
    T = len(ut)
    sampler = FactorLoadingParticleGibbs(n, p, T)
    
    #Initialize parameters (could be passed as arguments)
    theta = {
        'mu': np.random.normal(0.4, np.sqrt(2.0), size=n*p),
        'phi': np.random.normal(0.985, np.sqrt(0.001), size=n*p),
        'sigma': 1/np.random.gamma(20, 1/0.25, size=n*p)
    }
    
    # Initialize path
    current_path = np.random.normal(
        theta['mu'], 
        np.sqrt(theta['sigma']), 
        size=(T, n*p)
    )
    
    # iN ORDER TO Store samples
    samples = np.zeros((n_iterations, T, n*p))
    
    for i in tqdm(range(n_iterations), desc='Sampling', colour='blue'):
        #print(i)
        current_path = sampler.sample_path(ut, Xt, zt, zeta_t, theta, current_path)
        samples[i] = current_path
    return samples

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming you have the FactorLoadingParticleGibbs class from the previous code

# Load and prepare the data
def prepare_stock_data(returns_df, n_factors=1):
    """
    Prepare stock returns data for the factor loading model
    
    Parameters:
    returns_df: pandas DataFrame with daily returns (companies in columns, dates in index)
    
    Returns:
    tuple of (ut, Xt, zt, zeta_t) ready for the model
    """
    # Convert returns to numpy array
    ut = returns_df.values  # T x n matrix
    T, n = ut.shape
    print(T, n)
    
    # Create simple market factor (mean across all stocks)
    # You could replace this with a more sophisticated factor like SP500 returns
    zt = np.mean(ut, axis=1).reshape(-1, 1)  # T x 1 matrix
    # make zt a matrix of shape T x n_factors
    zt = np.repeat(zt, n_factors, axis=1)

    
    # Create dummy covariates (you can replace with actual features)
    Xt = np.ones((T, n, 1))  # Simple intercept-only covariates
    
    # Create mixing variables (can be adjusted based on your needs)
    zeta_t = np.ones(T)  # Simplest case: homoscedastic errors
    
    return ut, Xt, zt, zeta_t

# Run the model
def analyze_stock_returns(returns_df, n_iterations=1000, n_factors=1):
    """
    Analyze stock returns using Factor Loading Particle Gibbs
    
    Parameters:
    returns_df: pandas DataFrame with daily returns
    n_iterations: int, number of MCMC iterations
    n_factors: int, number of factors to use
    
    Returns:
    dict containing results and plots
    """
    # Prepare data
    ut, Xt, zt, zeta_t = prepare_stock_data(returns_df, n_factors)
    T, n = ut.shape
    p = n_factors
    
    # Run sampler
    samples = run_factor_loading_sampler(
        ut=ut,
        Xt=Xt,
        zt=zt,
        zeta_t=zeta_t,
        n=n,
        p=p,
        n_iterations=n_iterations
    )
    print(samples.shape)
    # Calculate mean factor loadings
    mean_loadings = np.mean(samples, axis=0)  # Average over iterations
    
    # Plot results
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    
    # Plot factor loadings for first 10 companies
    for i in range(min(4, n)):
        plt.plot(mean_loadings[:, i], 
                label=f"Company {returns_df.columns[i]}", 
                alpha=0.7)
    
    plt.title("Mean Factor Loadings Over Time")
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return {
        'samples': samples,
        'mean_loadings': mean_loadings,
        'companies': returns_df.columns
    }



if __name__ == '__main__':
    n = 100  # number of series
    p = 2   # number of factors
    T = 200 # time points
    k = 1    # number of covariates

    # Lets test on Synthetic data
    ut = np.random.randn(T, n)  # Transformed observations
    Xt = np.random.randn(T, n, k)  # Covariates
    zt = np.random.randn(T, p)  # Factors
    zeta_t = np.random.gamma(5, 1, T)  # Mixing variables

    # Run sampler
    samples = run_factor_loading_sampler(ut, Xt, zt, zeta_t, n, p, n_iterations=1000) # we use 100 iterations for 

    plt.figure(figsize=(8, 5))
    plt.style.use('dark_background')
    for i in range(1):
        for j in range(1):
            plt.plot(samples[j, :, i], label=f"Lambda_{i+1}", alpha=1, color='orange', linewidth=1.0)
    plt.title("Path simulations of Lambda_i")
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--', color='gray', linewidth=0.2)
    plt.show()

    # mean over n_iterations
    mean_samples = np.mean(samples, axis=0)

    # Plot the mean path samples
    plt.figure(figsize=(8, 4))
    plt.style.use('dark_background')
    for i in range(1):
        plt.plot(mean_samples[:, i], label=f"Lambda_{i+1}")
    plt.title("Mean path simulations of Lambda_i")
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.2)
    plt.show()

    #-------------------------
    #Load stock returns data
    #-------------------------
    returns_df = pd.read_csv('../Data/daily_equity_returns_recent.csv', index_col=0)
    returns_df.dropna(axis=0, inplace=True)  # Drop columns with missing data
    
    # Run analysis
    results = analyze_stock_returns(
        returns_df=returns_df,
        n_iterations=1000,  # Start with fewer iterations for testing
        n_factors=2
    )
    
    plt.show()
    
    # Access results
    mean_loadings = results['mean_loadings']
    samples = results['samples']
    companies = results['companies']
    