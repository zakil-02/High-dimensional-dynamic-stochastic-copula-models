# This file contains the implementation of the Particle Gibbs Sampler algorithm in order to draw paths of factor loadings lambda.

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional
import particles

class ParticleGibbsSampler:
    def __init__(self, 
                 T: int,
                 M: int = 100,  # Number of particles (paper uses M=100)
                 mu_prior: Tuple[float, float] = (0.4, 2.0),  # N(0.4, 2) prior for mu
                 phi_prior: Tuple[float, float] = (0.985, 0.001),  # N(0.985, 0.001) for phi
                 sigma_prior: Tuple[float, float] = (20, 0.25)):  # IG(20, 0.25) for sigma
        self.T = T  # Time series length
        self.M = M  # Number of particles
        
        # Prior parameters
        self.mu_prior = mu_prior
        self.phi_prior = phi_prior
        self.sigma_prior = sigma_prior
        
    def proposal_distribution(self, 
                            lambda_prev: float, 
                            ut: float, 
                            Xt: np.ndarray, 
                            zt: float, 
                            zeta_t: float, 
                            theta: dict) -> float:
        """
        Proposal distribution q(mabda_t|lambda_{t-1}, u_t, X_t, z_t, zeta_t, θ)
        Using a simple random walk proposal for demonstration
        """
        mu, phi, sigma = theta['mu'], theta['phi'], theta['sigma']
        mean = phi * lambda_prev + (1 - phi) * mu
        return np.random.normal(mean, np.sqrt(sigma))
    
    def calculate_importance_weights(self,
                                  particles: np.ndarray,
                                  prev_particles: np.ndarray,
                                  ut: float,
                                  Xt: np.ndarray,
                                  zt: float,
                                  zeta_t: float,
                                  theta: dict) -> np.ndarray:
        """
        Calculate importance weights for particles
        """
        mu, phi, sigma = theta['mu'], theta['phi'], theta['sigma']
        
        # Calculate transition density p(λ_t|λ_{t-1}, θ)
        transition_density = stats.norm.pdf(
            particles, 
            loc=phi * prev_particles + (1 - phi) * mu,
            scale=np.sqrt(sigma)
        )
        
        # Calculate likelihood p(u_t|z_t, X_t, lambda_t, zeta_t, θ)
        # Simplified likelihood for demonstration
        likelihood = stats.norm.pdf(ut, loc=particles * zt, scale=np.sqrt(zeta_t))
        
        # Calculate proposal density
        proposal_density = stats.norm.pdf(
            particles,
            loc=phi * prev_particles + (1 - phi) * mu,
            scale=np.sqrt(sigma)
        )
        
        # Return normalized weights
        weights = (likelihood * transition_density) / proposal_density
        return weights / np.sum(weights)
    
    def conditional_resampling(self,
                             particles: np.ndarray,
                             weights: np.ndarray,
                             reference_particle: float) -> np.ndarray:
        """
        Conditional multinomial resampling ensuring reference particle survives
        """
        M = len(particles)
        indices = np.zeros(M, dtype=int)
        
        # Always keep the reference particle
        indices[0] = 0
        particles[0] = reference_particle
        
        # Resample remaining M-1 particles
        remaining_indices = np.random.choice(
            M, size=M-1, p=weights, replace=True
        )
        indices[1:] = remaining_indices
        
        return particles[indices]
    
    def backwards_sampling(self,
                         particles_history: List[np.ndarray],
                         weights_history: List[np.ndarray],
                         theta: dict) -> np.ndarray:
        """
        Backward sampling pass to draw trajectory
        """
        T = len(particles_history)
        trajectory = np.zeros(T)
        
        # Sample last state
        last_idx = np.random.choice(self.M, p=weights_history[-1])
        trajectory[-1] = particles_history[-1][last_idx]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            mu, phi, sigma = theta['mu'], theta['phi'], theta['sigma']
            
            # Calculate backwards weights
            back_weights = weights_history[t] * stats.norm.pdf(
                trajectory[t+1],
                loc=phi * particles_history[t] + (1 - phi) * mu,
                scale=np.sqrt(sigma)
            )
            back_weights /= np.sum(back_weights)
            
            # Sample state
            idx = np.random.choice(self.M, p=back_weights)
            trajectory[t] = particles_history[t][idx]
            
        return trajectory
    
    def sample_path(self,
                   ut: np.ndarray,
                   Xt: np.ndarray,
                   zt: np.ndarray,
                   zeta_t: np.ndarray,
                   theta: dict,
                   reference_path: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run particle Gibbs sampler to draw a new path
        """
        particles_history = []
        weights_history = []
        
        # Initialize particles
        particles = np.random.normal(
            theta['mu'],
            np.sqrt(theta['sigma'] * 100),  # Inflated initial variance
            size=self.M
        )
        
        if reference_path is not None:
            particles[0] = reference_path[0]
            
        for t in range(self.T):
            # Store previous particles
            prev_particles = particles.copy()
            
            # Propose new particles (except reference)
            if reference_path is not None:
                particles[0] = reference_path[t]
            for m in range(1 if reference_path is not None else 0, self.M):
                particles[m] = self.proposal_distribution(
                    prev_particles[m], ut[t], Xt[t], zt[t], zeta_t[t], theta
                )
                
            # Calculate weights
            weights = self.calculate_importance_weights(
                particles, prev_particles, ut[t], Xt[t], zt[t], zeta_t[t], theta
            )
            
            # Conditional resampling
            if reference_path is not None:
                particles = self.conditional_resampling(particles, weights, reference_path[t])
            
            # Store particles and weights
            particles_history.append(particles.copy())
            weights_history.append(weights.copy())
            
        # Backward sampling
        return self.backwards_sampling(particles_history, weights_history, theta)

def run_particle_gibbs(ut: np.ndarray,
                      Xt: np.ndarray,
                      zt: np.ndarray,
                      zeta_t: np.ndarray,
                      n_iterations: int = 1000) -> np.ndarray:
    """
    Run the complete Particle Gibbs sampling procedure
    """
    T = len(ut)
    sampler = ParticleGibbsSampler(T)
    
    # Initialize parameters
    theta = {
        'mu': np.random.normal(0.4, np.sqrt(2.0)),
        'phi': np.random.normal(0.985, np.sqrt(0.001)),
        'sigma': 1/np.random.gamma(20, 1/0.25)
    }
    
    # Initialize path
    current_path = np.random.normal(theta['mu'], np.sqrt(theta['sigma']), size=T)
    
    # Store samples
    samples = np.zeros((n_iterations, T))
    
    # Run sampler
    for i in range(n_iterations):
        # Sample new path
        current_path = sampler.sample_path(ut, Xt, zt, zeta_t, theta, current_path)
        samples[i] = current_path
        
        # Update parameters (theta) this involve sampling from the full conditionals
        # (Not implemented here for brevity )
        
    return samples









# # Example usage
# # Example usage
# T = 1000  # Number of time points
# ut = np.random.randn(T)  # Your data
# Xt = np.random.randn(T)  # Your covariates
# zt = np.random.randn(T)  # Common factors
# zeta_t = np.random.gamma(5, 1, T)  # Mixing variables

# # Run sampler
# samples = run_particle_gibbs(ut, Xt, zt, zeta_t, n_iterations=1000)