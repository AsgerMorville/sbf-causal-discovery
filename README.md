# Smooth Backfitting Library for Causal Discovery

This is a small Python package that lets you generate data, fit a directed acyclic graph and evaluate the results using both the structural intervention distance and the structural Hamming distance.

## Quick Install

Ensure first that poetry is installed. Then run

```shell
# 1. Clone
 git clone https://github.com/AsgerMorville/sbf‑causal‑discovery.git
 cd sbf‑causal‑discovery

# 2. Install packages
 poetry install
```
You should now be able to run the scripts inside the /examples folder. To test this, run

```shell
poetry run python exampes/run_simulation.py
```

## Running the algorithm

Here is a simple example to get you started. 
We sample 100 IID observations from a model with 10 variables. The underlying causal structure is assumed to be sparse.
It shows the entire work flow: simulate → search → prune → inspect.

```Python
from simulation_utils.synthetic_data import SyntheticData
from graph_search.ordering_search import GreedyOrderingSearch
from sbf_core.regression import SBF_fitter
from graph_search.variable_selection import EdgePruner

########################################################
# 1. Simulation configuration and data generation
########################################################

# Set configuration for simulation
n = 100 # Number of samples
p = 10 # Number of observed variables
lmbda = 0.5 # Controls the sparsity of the estimated DAG. Higher lmbda gives estimates that are sparser.
length_scale = 0.4 # Controls the wigglyness of the structural equation functions
hetero_var = False # Set to true to simulate heterogeneous variance
dense_graph = 'never_dense' #never_dense: sparse setting. Set to 'always_dense' for dense setting.

# Generate DAG and samples
synth_data = SyntheticData(p=p, n=n, length_scale=length_scale, hetero_var=hetero_var, dense_graph=dense_graph)

########################################################
# 2. Fitting the fully connected DAG
########################################################

data = synth_data.X

# Use the smooth backfitting method as the fitter
fitter = SBF_fitter()

# Initialize the search method with which to perform the DAG search
model = GreedyOrderingSearch(fitter, data)

# Run the training phase.
model.train()

########################################################
# 3. Pruning the DAG
########################################################

# Prune the graph estimate
pruner = EdgePruner(G=model.G, X=data, lambda_par=lmbda, selection_method='fLassoMan')

# Run the pruning process
pruner.train()

########################################################
# 4. Inspect result
########################################################

print("Graph estimate: ", pruner.G.G)
print("True graph: ", synth_data.G.G)
```