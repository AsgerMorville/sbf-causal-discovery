import numpy as np

from sbf_core.f_lasso import fLasso


class EdgePruner:
    def __init__(self, G, X, lambda_par=1, selection_method='fLassoMan', bandwidth_method='Silverman'):
        self.G = G
        self.X = X
        self.lambda_par = lambda_par
        self.p = self.G.p
        self.selection_method = selection_method
        self.bandwidth_method = bandwidth_method

    def train(self):
        """
        This function performs the edge pruning of the graph
        """
        for j in range(self.p):
            # We take the j'th column in X, find its parent columns according to G
            # Then we find the relevant edges using backward selection
            parents = self.G.parents(j)
            if len(parents) == 0:
                continue
            no_of_parents = len(parents)

            # Construct Covariates and Y
            X_covariates = self.X[:, parents]
            Y = self.X[:, j]

            # Dictionary to keep track of indices
            index_dict = {i: k for k, i in zip(parents, range(no_of_parents))}

            # Figure out which variables are left after var selection
            if self.selection_method == 'fLassoMan':
                picked, deleted = fLassoSelection(Y=Y, X=X_covariates, lambda_par=self.lambda_par, bandwidth=self.bandwidth_method)
            if self.selection_method == 'fLassoAuto':
                picked, deleted = fLassoSelection(Y=Y, X=X_covariates, bandwidth=self.bandwidth_method)

            # Delete the relevant edges
            for l in deleted:
                self.G.removeEdge(index_dict[l], j)


def fLassoSelection(Y, X, lambda_par=None, bandwidth='Silverman'):
    if lambda_par is None:
        m_hat = fLasso(Y=Y, X=X, h=bandwidth)
    else:
        m_hat = fLasso(Y, X, lambda_par, h=bandwidth)
    est_vec = np.sum(np.power(m_hat, 2), axis=0)
    d = X.shape[1]
    eps = 0.001
    picked = []
    deleted = []
    for j in range(d):
        if est_vec[j] > eps:
            picked.append(j)
        else:
            deleted.append(j)
    return picked, deleted
