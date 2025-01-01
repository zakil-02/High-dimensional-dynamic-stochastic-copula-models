# High-dimensional-dynamic-stochastic-copula-models

This repository provides the implementation and experiments described in the paper:

**High Dimensional Dynamic Stochastic Copula Models** by Drew D. Creal and Ruey S. Tsay

The repository includes code, data processing scripts, and results for modeling and analyzing high-dimensional time-varying dependence structures across financial assets. The implementation uses Bayesian estimation techniques and particle Gibbs sampling to achieve efficient computation even in large-dimensional settings.

## Overview

Dynamic stochastic copula models provide a flexible framework for capturing time-varying dependencies in financial data. The models in this repository:

- Handle **high-dimensional dependence structures** efficiently using factor copula models.
- Incorporate **time-varying correlation matrices** to capture evolving relationships.
- Support **Gaussian**, **Student's t**, **grouped Student's t**, and **generalized hyperbolic copulas**.
- Utilize **Bayesian estimation** techniques with **particle Gibbs sampling** for accurate inference.

These models are particularly useful in analyzing financial datasets, such as credit default swaps (CDS) and equity returns, as demonstrated in the paper.

## Key Features

1. **Dynamic Copula Framework**:
   - Time-varying dependence parameters.
   - Flexibility to model heavy tails and extreme events.
2. **Efficient Estimation**:
   - Factor structure for scalability in high dimensions.
   - Bayesian methods leveraging particle Markov Chain Monte Carlo (PMCMC).
3. **Applications**:
   - Analysis of CDS and equity returns for 100 US corporations.
   - Evaluation of different copula families under various settings.

## Repository Structure

```
├── data/
├── scripts/
│   ├── copula_models.py     # Implementation of copula models
│   ├── bootstrapFilter.py # Bayesian estimation methods
│   ├── particle_gibbs.py    # Particle Gibbs sampler implementation
├── notebooks/
├── results/
│   ├── summary_statistics/  # Output from experiments
│   ├── plots/               # Generated plots for conditional correlations
├── README.md                
```

## Setup

### Prerequisites
- Python 3.8+
- Recommended libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `seaborn`
  - `statsmodels`

### Installation
Clone this repository:
```bash
git clone https://github.com/yourusername/high-dimensional-copula.git
cd high-dimensional-copula
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
Ensure the data files are in the `data/` directory. Use the provided scripts in `scripts/` to preprocess the data if required:
```bash
python scripts/preprocess_data.py
```

### Model Estimation
Run the main script to estimate copula models:
```bash
python scripts/copula_models.py
```

### Results and Visualization
Use the Jupyter notebooks in `notebooks/` to analyze and visualize results:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Results

Key findings from the experiments include:
- The grouped Student's t copula with time-varying factor loadings outperformed other models in capturing dependencies.
- Observed significant variation in tail dependencies across industries.
- Effective scalability demonstrated on a 200-dimensional dataset of CDS and equity returns.
  
## Citation
If you use this code or data, please cite the original paper:

```
@article{creal2015dynamiccopula,
  title={High Dimensional Dynamic Stochastic Copula Models},
  author={Creal, Drew D. and Tsay, Ruey S.},
  journal={Journal of Econometrics},
  year={2015}
}
```
