import numpy as np

from graph_search import graph


class GreedyOrderingSearch:
    def __init__(self, fitter, X):
        """
        p: number of observed variables
        fitter: a regressor object, used to estimate f_ik-function
        L: represents the potential gain of adding edge (i,j) in likelihood, (pxp) matrix.
        X: n x p matrix representing variables over which we find the causal relations
        """
        self.X = X
        self.p = self.X.shape[1]
        self.G = graph.Graph(self.p)
        self.fitter = fitter
        self.S = np.zeros(self.p)
        self.L1 = np.zeros((self.p, self.p))
        self.initializeS()
        self.initializeL()

    def initializeS(self):
        for j in range(self.p):
            self.S[j] = self.logLik(self.X[:,j], self.X[:,[]])
        return None

    def initializeL(self):
        np.fill_diagonal(self.L1, None)
        for j in range(self.p):
            self.columnUpdate(j)
        return None

    def train(self):
        """
        This function iterates the update-function until no more edges can be added to G
        """
        # Condition: Check if adding any edges results in a positive log lik increase
        while np.sum(self.L1 > 0) > 0:
            self.update()
        return None

    def update(self):
        """
        This function adds the edge to the graph G that gives biggest reduction in log-error
        """
        # 1. We identify the (i,j) tuple which yields biggest reduction in log error
        i, j = np.unravel_index(np.nanargmax(self.L1), self.L1.shape)

        # 2. We add the edge to G
        self.G.addEdge(i, j)

        # 3. We update S[j] to reflect that now the edge i->j has been added.
        self.S[j] = self.L1[i,j] + self.S[j]

        # 4. After adding an edge, some choices of L[l,k] will no longer be possible to be added for the next update.
        # These are due to either a) being added already, b) if k is a descendent of j (or j itself) then k cannot have
        # an edge to l if l is an ancestor of i (or i itself). Because then we would have a cycle.

        # Case a)
        self.L1[i, j] = None

        # Case b)
        for a in self.G.ancestors(i) + [i]:
            for d in self.G.descendants(j) + [j]:
                self.L1[d, a] = None

        # 5. We update column j of L to reflect that now the edge i->j has been added.
        self.columnUpdate(j)

        return None

    def columnUpdate(self,i):
        """
        Updates column i of the self.L1 matrix of score increases by adding edges
        """
        for k in range(self.p):
            if np.isnan(self.L1[k, i]) == False:
                newParents = self.G.parents(i) + [k]
                newScore = self.logLik(Y=self.X[:,i], X=self.X[:,newParents])
                self.L1[k,i] = newScore-self.S[i]
        return None

    def logLik(self,Y,X):
        fitted_values = self.fitter.fit(Y,X)
        sigma_hat = np.mean(np.power(Y - fitted_values, 2))
        return -np.log(sigma_hat)

    def check_graph_score(self,H):
        score = 0
        for j in range(self.p):
            if H.parents(j) == []:
                score += self.logLik(Y=self.X[:,j], X=self.X[:,[]])
            else:
                score += self.logLik(Y=self.X[:,j], X=self.X[:,H.parents(j)])
        return score