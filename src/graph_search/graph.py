import numpy as np


class Graph:
    def __init__(self, p, G=None, dense_graph='never_dense'):
        self.p = p
        self.dense_graph = dense_graph
        self.G = G if G is not None else self.G_init()

    def G_init(self):
        return np.zeros((self.p, self.p))

    def randomize(self):
        if self.dense_graph == 'always_dense':
            for i in range(self.p - 1):
                for j in range(i + 1, self.p):
                    self.G[i, j] = np.random.binomial(1, 6 / (self.p - 1))
        if self.dense_graph == 'mixed_dense':
            probability = np.random.uniform(low=2 / (self.p - 1), high=6 / (self.p - 1))
            for i in range(self.p - 1):
                for j in range(i + 1, self.p):
                    self.G[i, j] = np.random.binomial(1, probability)
        if self.dense_graph == 'never_dense':
            for i in range(self.p - 1):
                for j in range(i + 1, self.p):
                    self.G[i, j] = np.random.binomial(1, 2 / (self.p - 1))
        return None

    def parents(self, i):
        # This function returns the parents of the node i
        output = []
        for j in range(self.p):
            if self.G[j, i] == 1:
                output.append(j)
        return output

    def children(self, i):
        output = []
        for j in range(self.p):
            if self.G[i, j] == 1:
                output.append(j)
        return output

    def descendants(self, i):
        output = []
        for k in self.children(i):
            output.append(k)
            if self.children(k) != []:
                output += self.descendants(k)
        return list(set(output))

    def addEdge(self, i, j):
        self.G[i, j] = 1

    def removeEdge(self, i, j):
        self.G[i, j] = 0

    def flipEdge(self, i, j):
        self.addEdge(j, i)
        self.removeEdge(i, j)

    def getOrderingFromFullyConnected(self):
        return np.argsort(self.G.sum(axis=1))[::-1]

    def ancestors(self, i):
        output = []
        for k in self.parents(i):
            output.append(k)
            if self.parents(k) != []:
                # print(self.ancestors(k)[0])
                output += self.ancestors(k)
        return list(set(output))

    def fully_connected(self):
        """
        returns a fully connected DAG
        """
        X_graph = np.zeros((self.p, self.p))
        for j in range(self.p):
            if j > 0:
                X_graph[0:j, j] = 1
        return np_to_graph(X_graph)

    def tester_connected(self):
        X_graph = np.zeros((self.p, self.p))
        for j in range(self.p):
            X_graph[0:(j - 1), j] = 1
        return np_to_graph(X_graph)