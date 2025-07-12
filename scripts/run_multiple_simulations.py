import numpy as np

from simulation_utils.synthetic_data import SyntheticData
from graph_search.ordering_search import GreedyOrderingSearch
from sbf_core.regression import SBF_fitter
from graph_search.variable_selection import EdgePruner
from graph_search.graph_distance import struct_interv_dist, struct_hamming_dist

# Set configuration for simulation
n = 100 # Number of samples
p = 10 # Number of observed variables
lmbda = 0.5 # Controls the degree of sparsity of the estimated DAG. Higher lmbda gives estimates that are more sparse.
length_scale = 0.4 # Controls the wigglyness of the structural equation functions
hetero_var = True # Set to true to simulate heterogeneous variance
dense_graph = 'always_dense' #never_dense: sparse setting. Set to 'always_dense' for dense setting.

b = 100 # number of simulation runs

results = np.zeros((b,2))
for i in range(b):
    print(f"Simulation {i} out of {b}")
    # Generate DAG and samples
    synth_data = SyntheticData(p=p, n=n, length_scale=length_scale, hetero_var=hetero_var, dense_graph=dense_graph)

    ############################
    # Fitting the model
    ############################
    data = synth_data.X

    # Use the smooth backfitting method as the fitter
    fitter = SBF_fitter()

    # Initialize the search method with which to perform the DAG search
    model = GreedyOrderingSearch(fitter, data)

    # Run the training phase.
    model.train()

    # Prune the graph estimate
    pruner = EdgePruner(G=model.G, X=data, lambda_par=lmbda, selection_method='fLassoMan')

    # Run the pruning process
    pruner.train()

    results[i, 0] = struct_interv_dist(synth_data.G, pruner.G)
    results[i, 1] = struct_hamming_dist(synth_data.G, pruner.G)



# Final estimate:
print("Average SID: ", np.mean(results[:, 0]))
print("Average SHD: ", np.mean(results[:, 1]))





