import numpy as np
import numba as nb
from numpy.polynomial import Polynomial

from src.sbf_core.additive_function import AdditiveFunction

def initialize_SBF(X, Y):
    M = 100
    maxIter = 1000
    d = X.shape[1]
    Y_mean = Y.mean()
    Y_centered = Y - Y_mean
    dx = 1 / (M - 1)
    return M, maxIter, d, Y_mean, Y_centered, dx


def initialize_x(X, M):
    d = X.shape[1]
    n = X.shape[0]
    x_grid = np.linspace(np.zeros(d), np.ones(d), M)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_center = (X - X_min) / (X_max - X_min)
    return x_grid, X_center, X_min, X_max, n


def initialize_h(X):
    """
    We use Silvermans bandwidth.
    """
    n, d = X.shape
    q3, q1 = np.percentile(X, [75, 25], axis=0)
    IQR = q3 - q1
    m_vec = np.minimum(np.std(X, axis=0), IQR / 1.349)
    h = (0.9 / np.power(n, 1 / 5)) * m_vec

    # Check phat(x) > 0 for all x
    sorted_X = np.sort(X, axis=0)
    diff_array = 0.5 * np.diff(sorted_X, axis=0)
    min_h = np.max(diff_array, axis=0)
    return np.maximum(min_h, h)


def bandwidth(y, x):
    degree = 4
    n = y.shape[0]
    C_nu_p = 1.719/2
    p = Polynomial.fit(x, y, degree)
    coef = p.convert().coef
    m_hhat_sum = np.sum(np.power(2 * coef[2] + 6 * coef[3] * x + 12 * coef[4] * np.power(x, 2), 2))

    # Obtain sigma hat
    y_pred = p(x)
    raw_residuals = y - y_pred
    sigma_hat = (1/(n-1)) * np.sum(np.power(raw_residuals,2))
    # std_residuals = np.std(raw_residuals)
    # standardized_residuals = raw_residuals / std_residuals
    # sigma_hat_sq = np.sum(np.power(standardized_residuals, 2))
    h_est = C_nu_p * np.power((sigma_hat * (x.max() - x.min()) / m_hhat_sum), 1 / 5)

    # Check phat(x) > 0 for all x
    sorted_X = np.sort(x)
    diff_array = 0.5 * np.diff(sorted_X, axis=0)
    min_h = np.max(diff_array, axis=0)

    return np.maximum(min_h, h_est)

def bandwidth_0_05(y, x):
    h_est = 0.05

    # Check phat(x) > 0 for all x
    sorted_X = np.sort(x)
    diff_array = 0.5 * np.diff(sorted_X, axis=0)
    min_h = np.max(diff_array, axis=0)

    return np.maximum(min_h, h_est)

def bandwidth_0_10(y, x):
    h_est = 0.10

    # Check phat(x) > 0 for all x
    sorted_X = np.sort(x)
    diff_array = 0.5 * np.diff(sorted_X, axis=0)
    min_h = np.max(diff_array, axis=0)

    return np.maximum(min_h, h_est)


def initialize_h_2(Y, X):
    n, d = X.shape
    h = np.zeros(d)
    for j in range(d):
        h[j] = bandwidth(Y, X[:, j])
    return h

def initialize_h_3(Y, X):
    n, d = X.shape
    h = np.zeros(d)
    for j in range(d):
        h[j] = bandwidth_0_05(Y, X[:, j])
    return h

def initialize_h_4(Y, X):
    n, d = X.shape
    h = np.zeros(d)
    for j in range(d):
        h[j] = bandwidth_0_10(Y, X[:, j])
    return h


@nb.jit('(float64[:,], float64)', fastmath=True)
def integrate_0_1(vector, dx):
    return (2 * vector.sum() - vector[0] - vector[-1]) * dx / 2


def SBF(Y, X, h='Silverman'):
    """
    Returns the fitted values.
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    m_hat = fLasso(Y=Y, X=X, lambda_par=0, h=h)
    M = m_hat.shape[0]
    fin_x_grid = np.linspace(X_min, X_max, M)
    additive_function = AdditiveFunction(fin_x_grid, m_hat, Y.mean())
    return additive_function.predict(X)


def SBF_Lasso(Y, X, lmbda):
    """
    Returns the fitted values.
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    m_hat = fLasso(Y=Y, X=X, lambda_par=lmbda)
    M = m_hat.shape[0]
    fin_x_grid = np.linspace(X_min, X_max, M)
    additive_function = AdditiveFunction(fin_x_grid, m_hat, Y.mean())
    return additive_function.predict(X)


def standardize(X):
    """
    Standardizes the columns of X to have mean 0 and variance 1.
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # return (X - np.mean(X, axis=0)) / np.sqrt(np.sum(np.power(X, 2), axis=0))


def fLasso(Y, X, lambda_par=None, h='Silverman'):
    """
    Input Y: n x 1 np array
    M: number of grid points of [0,1]
    dx: distance between grid points
    d: dimension of X
    maxIter: maximum number of iterations for the SBF loop
    """
    # Initialize parameters
    M, maxIter, d, Y_mean, Y_center, dx = initialize_SBF(X, Y)

    # Initialize, transform x-point grid:
    x_grid, X_center, X_min, X_max, n = initialize_x(X, M)

    # Initialize bandwidth h if h is not provided
    if h == 'Silverman':
        h = initialize_h(X_center)
    elif h == 'Polynomial':
        h = initialize_h_2(Y, X_center)
    elif h == '0_05':
        h = initialize_h_3(Y, X_center)
    elif h == '0_10':
        h = initialize_h_4(Y, X_center)
    else:
        ValueError('Bandwidth method not implemented')

    # Precompute tables
    Kh_table = Kh_table_generator(X_center, x_grid, h)
    p_hat_tab = phat_table_generator(X_center, x_grid, Kh_table)
    p_hat_tab2 = phat2_table_generator(X_center, x_grid, Kh_table)
    f_hat_tab = fhat_table_generator(Y_center, X_center, x_grid, p_hat_tab, Kh_table)

    # Optimization loop. If lambda par is not provided: pick by minimizing BIC
    m_hat = full_opt_loop(p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par, maxIter)

    return m_hat

def integrate2d(array, arrLength, dx):
    arrSum = 4 * np.sum(array)
    for l in range(arrLength):
        arrSum -= 2 * array[0, l] + 2 * array[-1, l] + 2 * array[l, 0] + 2 * array[l, -1]
    return arrSum * dx ** 2 / 4



@nb.jit('(float64[:,:],float64[:,:], int64)', fastmath=True)
def checkconv(m_old, m_new, d):
    for j in range(d):
        if np.sum(np.power(m_old[:, j] - m_new[:, j], 2)) / (np.sum(np.power(m_old[:, j], 2)) + 0.00001) > 0.00001:
            return False
    return True


@nb.jit('(float64[:,:],float64[:,:], float64[:,:,:,:], float64[:,:], float64, float64)', fastmath=True)
def opt_loop(m_hat, p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par):
    M, d = m_hat.shape
    normalizing_const = np.ones(d)
    for j in range(d):
        Pi_minus_j = np.zeros(M)
        for l in range(M):
            integral_sum = 0
            for k in range(d):
                if k != j:
                    integrand = m_hat[:, k] * p_hat_tab2[l, :, j, k]
                    integral_sum += integrate_0_1(integrand, dx)
            Pi_minus_j[l] = f_hat_tab[l, j] - integral_sum / (p_hat_tab[l, j])
        # Calculate norm of Pi_minus_j:
        integrand = np.power(Pi_minus_j, 2) * p_hat_tab[:, j]
        Pi_minus_j_norm = np.sqrt(integrate_0_1(integrand, dx))
        normalizing_const[j] = np.maximum(0, 1 - lambda_par / Pi_minus_j_norm)
        m_hat[:, j] = normalizing_const[j] * Pi_minus_j

    return m_hat


@nb.jit('(float64[:,:],float64[:,:,:,:], float64[:,:], float64, float64, float64)', fastmath=True)
def full_opt_loop(p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par, maxIter):
    M, d = p_hat_tab.shape
    m_hat = np.zeros((M, d))
    for r in range(maxIter):
        m_old = np.copy(m_hat)
        m_hat = opt_loop(m_hat, p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par)
        if checkconv(m_old, m_hat, d):
            # print("convergence at: iteration=", r)
            break
    return m_hat


def Kh_table_generator(X, x_points, h):
    output = ker(x_points[:, None, :] - X, h)
    output /= np.trapz(output, x_points[:, np.newaxis], axis=0)
    return output


def phat_table_generator(X, x_points, Kh_table):
    n = X.shape[0]
    return np.sum(Kh_table, axis=1) / n


def phat2_table_generator(X, x_points, Kh_table):
    n = X.shape[0]
    output = np.tensordot(Kh_table, Kh_table, axes=(1, 1)) / n
    output = np.transpose(output, axes=(0, 2, 1, 3))
    return output


def fhat_table_generator(Y, X, x_points, phat_table, Kh_table):
    # Output is (M,d)
    n = X.shape[0]
    output = np.tensordot(Y, Kh_table, axes=(0, 1))
    return output / (n * phat_table)


def NW_estimator(Y, X, phat_table, Kh_table):
    # Output is (M,d)
    n = X.shape[0]
    output = np.tensordot(Y, Kh_table, axes=(0, 1))
    return output / (n * phat_table)


def epan(x):
    mask = np.abs(x) < 1
    result = np.zeros_like(x)
    result[mask] = (3 / 4) * (1 - np.power(x[mask], 2))
    return result


def ker(x, h):
    xnew = np.zeros(x.shape)
    d = x.shape[2]
    for j in range(d):
        xnew[:, :, j] = x[:, :, j] / h[j]
    return (1 / h) * epan(xnew)



