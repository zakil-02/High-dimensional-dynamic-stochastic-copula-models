import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly
import sklearn
import particles
from particles import state_space_models as ssm
from particles import distributions as dists

from CopulaSSM import CopulaSSM

# Parameters for the model
n = 3
p = 5
k = 1

# Instantiate the model
copula_model = CopulaSSM(n=n, p=p, k=k)

print("Model instantiated successfully.")

# Simulate data
sim_data = copula_model.simulate(200)  # Simulate 200 samples
u, A = sim_data

print("Data simulated successfully.")

# Plot the data
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
colors = sns.color_palette("husl", n)
for i in range(n):
    plt.plot([u[j][0,i] for j in range(len(u))], label=f'u_{i+1}', linewidth=1.0, color=colors[i])
plt.title("Simulated data")
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.savefig('./simulated_data.png')

plt.show()


plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
for i in range(n):
    plt.plot([A[j][0,i] for j in range(len(A))], label=f'Λ_{i+1}', linewidth=0.9, color=colors[i])
plt.title("./Latent state")
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.savefig('./latent_state.png')
plt.show()