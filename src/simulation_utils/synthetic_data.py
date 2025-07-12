import numpy as np
from graph_search.graph import Graph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy import interpolate

class SyntheticData:
    # This class generates a
    def __init__(self, p, n, length_scale=1, M=100, hetero_var=False, dense_graph='never_dense'):
        self.p = p
        self.n = n
        self.hetero_var = hetero_var
        self.dense_graph = dense_graph
        self.G = Graph(p,dense_graph=self.dense_graph)
        self.G.randomize()
        self.lengthscale = length_scale
        self.M = M
        self.X = self.generate_data()

    def generate_data(self):
        output = np.zeros((self.n, self.p))
        # np.random.normal(scale=1, size=(self.n,self.p))
        # Loop for generating functions, observations
        for j in range(self.p):
            if self.hetero_var:
                #sigma_j_squared = 1
                sigma_j_squared = np.random.uniform(low=0.4, high=0.8)
                #alpha_rand = np.random.uniform(low=1, high=2)
                alpha_rand = 1
            else:
                sigma_j_squared = np.random.uniform(low=0.4, high=0.8)
            # Generate noise for j'th component
            noise = np.random.normal(scale=1, size=self.n) * np.sqrt(sigma_j_squared)

            # Generate the functions relating Xj to its parents
            cond_mean = np.zeros(self.n)
            var_vec = np.ones(self.n)
            #var_vec = np.zeros(self.n)
            for k in self.G.parents(j):
                # Find the interval domain of Xk
                xkmin = output[:, k].min()
                xkmax = output[:, k].max()

                # func_kj is a GP-obj, i.e. a 1d gaussian process defined on the given interval
                func_kj = GP(xkmin, xkmax, ls=self.lengthscale, M=self.M)
                #print("THIS IS f(xkmin): ", func_kj.f(xkmin))
                #print("THIS IS random: ", np.random.normal())
                if self.hetero_var:
                    unif = np.random.uniform(0,1)
                    if unif > 0.7:
                        #var_vec *= g(output[:, k])
                        var_vec += g3(output[:, k],alpha_rand)
                cond_mean += func_kj.f(output[:, k])

                """
                funcji = GP(xjmin, xjmax, ls=self.lengthscale, M=self.M)
                if self.hetero_var == True:
                    x *= g(output[:, j])
                x += funcji.f(output[:, j])"""
            if len(self.G.parents(j)) == 0:
                output[:, j] = cond_mean + noise
            else:
                output[:, j] = cond_mean + var_vec * noise

        return output

class GP:
    def __init__(self, xmin, xmax, ls, M):
        self.lengthscale = ls
        self.x_min = xmin
        self.x_max = xmax
        self.M = M
        self.f = self.gen_func()

    def gen_func(self):
        #print("THIS IS LENGTHSCALE: ", self.lengthscale)
        kernel = 1.0 * RBF(length_scale=self.lengthscale, length_scale_bounds=(1e-1, 10.0))
        #kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-1, 10.0))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=None)
        return self.gen_gpr_samples(gpr, 1)

    def gen_gpr_samples(self, gpr_model, n_samples):
        x = np.linspace(self.x_min, self.x_max, self.M)
        X = x.reshape(-1, 1)
        y_samples = np.squeeze(gpr_model.sample_y(X, n_samples, random_state=None))
        interp = interpolate.interp1d(x, y_samples, kind="linear")
        return interp

def g3(x,alpha):
    return 1 / (1 + np.exp(-x + 2)) - 1 / (1 + np.exp(-x - 2)) + alpha