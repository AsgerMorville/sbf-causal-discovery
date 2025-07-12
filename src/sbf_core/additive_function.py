import numpy as np


class AdditiveFunction:
    def __init__(self, x_points, m_points, y_mean):
        self.x_points = x_points
        self.m_points = m_points
        self.d = self.x_points.shape[1]
        self.M = x_points.shape[0]
        self.y_mean = y_mean

    def predict(self, X):
        n = X.shape[0]
        m_eval_array = np.zeros((n, self.d))
        for i in range(n):
            dif_arr = np.absolute(self.x_points - X[i, :])
            ind_tuple = dif_arr.argmin(axis=0)
            m_eval_array[i, :] = self.m_points[ind_tuple, np.arange(self.d)]
        return m_eval_array.sum(axis=1) + self.y_mean