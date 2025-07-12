import numpy as np

from sbf_core.f_lasso import SBF


class SBF_fitter:
    def fit(self, Y, X):
        if X.shape[1] == 0:
            n = Y.shape[0]
            return Y.mean() * np.zeros(n)
        else:
            return SBF(Y, X)