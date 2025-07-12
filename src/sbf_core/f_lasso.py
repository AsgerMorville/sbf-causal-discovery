import numpy as np
import numba as nb
from numpy.polynomial import Polynomial


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


def softThreshold(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)


def SS(x, alpha):
    return np.where(x > alpha, x - alpha, np.where(x < alpha, x + alpha, 0))


def S(x, alpha):
    nn = x.shape[0]
    output = np.zeros(nn)
    for i in range(nn):
        if x[i] > alpha:
            output[i] = x[i] - alpha
        elif x[i] < -alpha:
            output[i] = x[i] + alpha
        else:
            output[i] = 0
    return output


def OGLassoWrapper(Y, X, lmbda=None):
    if lmbda is not None:
        return OGLasso(Y, X, lmbda)
    else:
        lmbdaGridSize = 50
        lmbdaGrid = np.geomspace(0.0001, 2, num=lmbdaGridSize)
        stdX = standardize(X)
        # Calculate BIC values for each lambda
        BIC_values = np.zeros(lmbdaGridSize)
        for k in range(lmbdaGridSize):
            beta_hat = OGLasso(Y, X, lmbdaGrid[k])
            BIC_values[k] = BIC_estimate_OG(Y, beta_hat, stdX)

        bestLmbda = lmbdaGrid[np.argmin(BIC_values)]
        return OGLasso(Y=Y, X=X, lmbda=bestLmbda)


def OGLasso(Y, X, lmbda):
    if X.shape[1] == 0:
        raise TypeError("X should not be empty")
    if np.array_equal(X[:, 0], np.ones(X.shape[0])):
        raise TypeError("X should not come with intercept column")

    X = standardize(X)
    Y = Y - Y.mean()

    n = X.shape[0]
    d = X.shape[1]
    eps = 0.00001
    maxIter = 10000
    eigvals, eigvecs = np.linalg.eigh(X.T @ X / n)
    L = np.max(eigvals)
    beta_hat = np.zeros(d)
    for k in range(maxIter):
        # print(k)
        beta_old = np.copy(beta_hat)
        # beta_hat = S(beta_old-1/(n*L)*(X.T@X@beta_old - X.T@Y), lmbda/L)
        beta_hat = beta_step(beta_old, X, Y, L, lmbda)
        if np.linalg.norm(beta_old - beta_hat, 2) < eps:
            # print("convergence at iteration :", k)
            break
    return beta_hat


def beta_step(beta_old, X, Y, L, lmbda):
    n = X.shape[0]
    print(L)
    return S(beta_old - 1 / (n * L) * (X.T @ X @ beta_old - X.T @ Y), lmbda / L)

def NewtonSolver(Y, X, theta, lmbda):
    m = X.shape[1]
    n = Y.shape[0]
    for l in range(m):
        def f(theta_l):
            return 0.5 * (1/n)  * np.sum((Y - X@theta + X[:,l]*theta[l] - X[:,l]*theta_l) ** 2) \
                + lmbda * np.power(np.sum(np.power(theta,2))-theta[l]**2 + theta_l**2,0.5)
        x0 = np.array([0])
        theta_l = minimize(f, x0, method='BFGS')
        theta[l] = theta_l.x[0]
    return theta

def beta_l_update(Y, index, W_list, beta_list, lmbda):
    # Check if solution is zero
    m = len(W_list)
    n = Y.shape[0]
    beta_l_len = beta_list[index].shape[0]

    # We check first if the beta is zero.
    residual = np.copy(Y)
    for j in range(m):
        if j is not index:
            residual -= W_list[j] @ beta_list[j]
    if np.sqrt((1/n) * np.sum(np.power(W_list[index].T@residual,2))) < lmbda:
        return np.zeros(beta_l_len)
    else:
        # In this case, the solution of beta_l is not zero.
        new_beta = NewtonSolver(Y=Y, X=W_list[index], theta=beta_list[index], lmbda=lmbda)
        return new_beta

def betaUpdateMultinomial(old_beta, W, Y, lmbda):
    """
    m is the number of groups for the covariate W.
    :param old_beta: list of length m. Each element contains np array with current beta estimates
    :param W: list of length m. The STANDARDIZED inputs. Each element of the list corresponds to one grouping.
    :param Y: np.vector of length n.
    :param lmbda: lasso parameter for penalizing.
    :return: updated old_beta after ONE full cycle of coordinate descent. Repeatedly calling this function
    should lead to convergence under suitable conditions
    """
    m = len(W)
    for l in range(m):
        old_beta[l] = beta_l_update(Y=Y, index=l, W_list=W, beta_list=old_beta, lmbda=lmbda)
    return old_beta



def standardize(X):
    """
    Standardizes the columns of X to have mean 0 and variance 1.
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # return (X - np.mean(X, axis=0)) / np.sqrt(np.sum(np.power(X, 2), axis=0))


def fPLSBF2(Y, X, Z, Y_type=None, lmbda=None, h=None):
    if X.shape[1] == 0:
        raise TypeError("X should not be empty")
    if Z.shape[1] == 0:
        raise TypeError("Z should not be empty")
    # We assume that X comes without its intercept
    if np.array_equal(X[:, 0], np.ones(X.shape[0])):
        raise TypeError("X should not come with intercept column")

    gridData = GridDataMixed(Y=Y, X=X, Z=Z, Ytype=Y_type, h=h)

    # The following L-constant is needed for the lasso
    eigvals, eigvecs = np.linalg.eigh(gridData.stdX.T @ gridData.stdX / gridData.n)
    L = np.max(eigvals)

    # Now, we fit the function. If the lambda-parameter is unspecified, we find it by
    # minimizing a BIC criterion.
    if lmbda is not None:
        m_hat, beta_hat = LassoOptLoop(gridData, L, lmbda)
    else:
        lmbdaGridSize = 25
        lmbdaGrid = np.geomspace(0.01, 1, num=lmbdaGridSize)

        # Calculate BIC values for each lambda
        BIC_values = np.zeros(lmbdaGridSize)
        for k in range(lmbdaGridSize):
            m_hat, beta_hat = LassoOptLoop(gridData=gridData, L=L, lmbda=lmbdaGrid[k])
            BIC_values[k] = BIC_estimate2(mHat=m_hat, betaHat=beta_hat, gridData=gridData)
            # BIC_values[k] = log_lik_BIC(mHat=m_hat, betaHat=beta_hat, gridData=gridData)

        bestLmbda = lmbdaGrid[np.argmin(BIC_values)]
        m_hat, beta_hat = LassoOptLoop(gridData, L, bestLmbda)
    return m_hat, beta_hat


# def LassoOptLoop(gridData, L, lmbda):
#    return GurobiOptLoop(gridData, lmbda)

def betaUpdate(beta_output, stdX, Y_tilde, lmbda):
    """
    This function updates beta_old using coordinate descent.
    """
    d_D = beta_output.shape[0]
    n = Y_tilde.shape[0]
    for j in range(d_D):
        """thresh = 0
        print("Check")
        for i in range(n):
            yitilde = 0
            for k in range(d_D):
                yitilde += stdX[i,k] * beta_output[k]
            yitilde -= stdX[i,j] * beta_output[j]
            thresh += stdX[i,j] * (Y_tilde[i] - yitilde)
        beta_output[j] = softThreshold(thresh, lmbda)"""
        # subBeta = np.delete(beta_output, j, axis=0)
        # subX = np.delete(stdX, j, axis=1)
        # beta_output[j] = softThreshold(np.dot(stdX[:,j], Y_tilde-subX@subBeta), lmbda)
        # beta_output[j] = softThreshold((beta_output[j] + np.dot(stdX[:,j], Y_tilde-stdX@beta_output)/n), lmbda * n)
        beta_output[j] = softThreshold(beta_output[j] + (np.dot(stdX[:, j], Y_tilde - stdX @ beta_output) / n),
                                       lmbda)
    return beta_output


def betaUpdateMultinomial(old_beta, W, Y, lmbda):
    """
    m is the number of groups for the covariate W.
    :param old_beta: list of length m. Each element contains np array with current beta estimates
    :param W: list of length m. The STANDARDIZED inputs. Each element of the list corresponds to one grouping.
    :param Y: np.vector of length n.
    :param lmbda: lasso parameter for penalizing.
    :return: updated old_beta after ONE full cycle of coordinate descent. Repeatedly calling this function
    should lead to convergence under suitable conditions
    """
    m = len(W)
    for l in range(m):
        old_beta[l] = beta_l_update(Y=Y, index=l, W_list=W, beta_list=old_beta, lmbda=lmbda)
    return old_beta


def beta_l_update(Y, index, W_list, beta_list, lmbda):
    # Check if solution is zero
    m = len(W_list)

    # We check first if the beta is zero.
    first_vec = Y
    for j in range(m):
        if j is not index:
            first_vec -= W_list[j] @ beta_list[j]
    if np.dot(W_list[index], first_vec) < lmbda:
        return np.zeros(m)
    else:
        # In this case, the solution of beta_l is not zero.
        X = W_list[index]
        k = X.shape[1]
        for t in range(k):
            # For each coordinate we optimize
            beta_list[index][t] = NewtonSolver(Y=Y, X=X, theta=beta_list[index], lmbda=lmbda)

from scipy.optimize import minimize
def NewtonSolver(Y, X, lmbda, theta):
    """
    This function goes through one cycle of coordinate descent for the particular group
    :param Y:
    :param X:
    :param lmbda:
    :param theta:
    :return:
    """
    m = len(X)
    output = np.zeros(m)
    for l in range(m):
        def f(theta_l):
            return 0.5 * np.sum((Y - X@theta + X[:,l]*theta[l] - X[:,l]*theta_l) ** 2) \
                   + lmbda * np.power(np.sum(np.power(theta,2))-theta[l]**2 + theta_l**2,0.5)
        x0 = np.array([0])
        theta_l = minimize(f, x0, method='BFGS', options={'approx_grad': True})
        output[l] = theta_l
    return

def LassoOptLoop(gridData, L, lmbda):
    beta_hat = np.zeros(gridData.d_d)
    m_hat = np.zeros((gridData.M, gridData.d_c))
    for l in range(gridData.maxIter):
        beta_old = np.copy(beta_hat)
        m_old = np.copy(m_hat)

        # 1. The function update part
        # First, we update the marginal expectation function
        # Here, we simply replace the Ycenter with Ycenter - stdX.T@beta_hat
        marginalNW = gridData.fHat - np.sum(gridData.X_NW_table * beta_hat.reshape(1, 1, gridData.d_d), axis=2)
        # marginalNWCentered = marginalNW - marginalNW.mean()
        m_hat = opt_loop(m_hat, gridData.pHat, gridData.pHat2, marginalNW, gridData.dx, lmbda)
        # m_hat = full_opt_loop(gridData.pHat, gridData.pHat2, marginalNW,gridData.dx, lmbda,gridData.maxIter)

        # 2. The betahat update part
        # We need the fitted values implied by the current m_hat values
        # fin_z_grid = np.linspace(gridData.Z_min, gridData.Z_max, gridData.M)
        # additive_function = AdditiveFunction(x_points=fin_z_grid, m_points=m_hat, y_mean=gridData.Y.mean())
        # fitted = additive_function.predict(gridData.Z)

        # Construct the Ytilde values which are the original Y-values subtracted the function values
        Y_tilde = np.copy(gridData.Y)
        for i in range(gridData.n):
            for j in range(gridData.d_c):
                Y_tilde[i] -= integrate_0_1(m_hat[:, j] * gridData.KhTab[:, i, j], gridData.dx)

        Y_tilde = Y_tilde - gridData.Y.mean()

        # Do a one step update with the thresholding function
        # beta_hat = beta_step(beta_old, gridData.stdX, Y_tilde, L, lmbda)
        beta_hat = betaUpdate(beta_old, gridData.stdX, Y_tilde, lmbda)
        # beta_hat = OGLasso(Y_tilde,gridData.stdX, lmbda)

        if checkconv(m_old, m_hat, gridData.d_c) and np.linalg.norm(beta_old - beta_hat, 2) < 0.0001:
            # print("convergence at: iteration=", l)
            break
        if l > gridData.maxIter - 2:
            print("no convergence")
    return m_hat, beta_hat


def xzSplitCovariateList(X_list, X_type):
    p = len(X_list)
    if p == 0:
        raise ValueError("X_list needs to have at least one element")

    n = X_list[0].shape[0]
    W_part = []
    Z_part = np.empty((n,0))
    for i in range(p):
        if X_type[i] == b'cont':
            Z_part = np.hstack(Z_part, X_list[i])
        elif X_type[i] == b'disc':
            W_part.append(X_list[i])
    return W_part, Z_part



def LinearGroupLasso(Y, W_list, lambda_par):
    beta_hat = [np.zeros(W_list[i].shape[1]) for i in range(len(W_list))]
    for i in range(30):
        beta_hat = betaUpdateMultinomial(beta_old=beta_hat, W=W_list, Y=Y, lmbda=lambda_par)
    return beta_hat



def fLassoMultinomial(Y, X_list, X_type, lambda_par):
    """
    This function dispatches the fLasso methods.
    :param Y: np array of dimension 1 or 2. These are the responses. If the response is continuous then Y.ndim = 1.
    If the response is categorical, then Y.ndim = 2.
    :param X_list: A list of the covariates of length d, where d may be zero.
    :param X_type:
    :param lambda_par:
    :return: It returns the fitted !coefficients/functions!. So if X_list is an empty list, and empty list is
    returned.
    """
    numberOfCovariates = len(X_list)

    output = []
    for j in range(Y.shape[1]):
        if numberOfCovariates == 0:
            output.append([])
        else:
            # Some covariates are present. We split the covariates into the continuous ones and the categorical ones.
            W_part, Z_part = xzSplitCovariateList(X_list, X_type)
            # Now W_part is a list consisting of the categorical indices, and Z_part is a numpy array of the continuous

            if W_part.shape[1] == 0:
                # No discrete indices present.
                mhat0 = fLasso(Y=Y[:,j], X=Z_part, lambda_par=lambda_par)
                betahat0 = np.empty(0)
            elif Z_part.shape[1] == 0:
                # No continuous covariates present.
                betahat0 = LinearGroupLasso(Y=Y[:,j], W_list = W_part, lambda_par=lambda_par)
                # betahat0 is now a list with len(betahat0) = len(W_part).
                mhat0 = np.empty((0,0))
            else:
                betahat0, mhat0 = groupSBFLasso(Y=Y[:,j], W=W_part, Z=Z_part, lambda_par=lambda_par)
                # betahat0 is a list with len(betahat0) = len(W_part).
                # mhat0 is a n x d numpy array, where d is the number of continuous covariates.

            # Next, we fix betahat and mhat so that they become lists of length p
            contCounter = 0
            discCounter = 0
            betaHat = []
            mHat = []
            for i in range(numberOfCovariates):
                if X_type[i] == b'cont':
                    mHat.append(mhat0[:,contCounter])
                    betaHat.append([])
                elif X_type[i] == b'disc':
                    mHat.append([])
                    betaHat.append(betahat0[discCounter])
            output.append([betaHat, mHat])
    return output

def groupSBFLasso(Y, W, Z, lambda_par):
    """
    Y is a vector of length n
    W is a list that is of length at least one
    Z is a numpy array n x d where d > 0
    Return: tuple (betahat, mhat) where betahat is a list of same length as W, mhat M x d array of estimated functions
    """
    if X.shape[1] == 0:
        raise TypeError("X should not be empty")
    if Z.shape[1] == 0:
        raise TypeError("Z should not be empty")
    # We assume that X comes without its intercept
    if np.array_equal(X[:, 0], np.ones(X.shape[0])):
        raise TypeError("X should not come with intercept column")

    gridData = GridDataMixed(Y=Y, X=X, Z=Z, Ytype=Y_type, h=h)

    # The following L-constant is needed for the lasso

    # Now, we fit the function. If the lambda-parameter is unspecified, we find it by
    # minimizing a BIC criterion.
    if lmbda is not None:
        m_hat, beta_hat = GroupLassoOptLoop(gridData, lmbda)
    return m_hat, beta_hat

def GroupLassoOptLoop(gridData, L, lmbda):
    beta_hat = np.zeros(gridData.d_d)
    m_hat = np.zeros((gridData.M, gridData.d_c))
    for l in range(gridData.maxIter):
        beta_old = np.copy(beta_hat)
        m_old = np.copy(m_hat)

        # 1. The function update part
        # First, we update the marginal expectation function
        # Here, we simply replace the Ycenter with Ycenter - stdX.T@beta_hat
        marginalNW = gridData.fHat - np.sum(gridData.X_NW_table * beta_hat.reshape(1, 1, gridData.d_d), axis=2)
        # marginalNWCentered = marginalNW - marginalNW.mean()
        m_hat = opt_loop(m_hat, gridData.pHat, gridData.pHat2, marginalNW, gridData.dx, lmbda)
        # m_hat = full_opt_loop(gridData.pHat, gridData.pHat2, marginalNW,gridData.dx, lmbda,gridData.maxIter)

        # 2. The betahat update part
        # Construct the Ytilde values which are the original Y-values subtracted the function values
        Y_tilde = np.copy(gridData.Y)
        for i in range(gridData.n):
            for j in range(gridData.d_c):
                Y_tilde[i] -= integrate_0_1(m_hat[:, j] * gridData.KhTab[:, i, j], gridData.dx)

        Y_tilde = Y_tilde - gridData.Y.mean()

        # Do a one step update with the thresholding function
        # beta_hat = beta_step(beta_old, gridData.stdX, Y_tilde, L, lmbda)
        #beta_hat = betaUpdate(beta_old, gridData.stdX, Y_tilde, lmbda)
        # beta_hat = OGLasso(Y_tilde,gridData.stdX, lmbda)
        beta_hat = betaUpdateMultinomial(old_beta=beta_old, W=gridData.stdW, Y= Y_tilde, lmbda=lmbda)

        if checkconv(m_old, m_hat, gridData.d_c) and np.linalg.norm(beta_old - beta_hat, 2) < 0.0001:
            # print("convergence at: iteration=", l)
            break
        if l > gridData.maxIter - 2:
            print("no convergence")
    return m_hat, beta_hat



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
    if lambda_par is not None:
        m_hat = full_opt_loop(p_hat_tab, p_hat_tab2, f_hat_tab, dx, lambda_par, maxIter)
    else:
        no_of_gridpoints = 50
        lambda_grid = np.geomspace(0.001, 2, num=no_of_gridpoints)

        # Calculate BIC values for each lambda
        BIC_values = [BIC_estimate(full_opt_loop(p_hat_tab, p_hat_tab2, f_hat_tab, dx, lam, maxIter),
                                   Y_center, p_hat_tab, p_hat_tab2, f_hat_tab, h, dx) for lam in lambda_grid]

        # Find best lambda value and return associated m_hat
        best_lambda = lambda_grid[np.argmin(BIC_values)]
        m_hat = full_opt_loop(p_hat_tab, p_hat_tab2, f_hat_tab, dx, best_lambda, maxIter)
        print("BEST LAMBDA: ", best_lambda)
    # return m_hat, Kh_table, p_hat_tab, f_hat_tab, p_hat_tab2
    return m_hat


@nb.jit('(float64[:,:],float64[:], float64[:,:], float64[:,:,:,:], float64[:,:], float64[:], float64)', fastmath=True)
def BIC_estimate(m_hat, Y_center, p_hat_tab, p_hat_tab2, f_hat_tab, h, dx):
    M, d = m_hat.shape
    n = Y_center.shape[0]

    # First term: log( 0.5 * (T1 + T2 - 2 * T3) )

    # T1 is the Y-term
    T1 = np.sum(np.power(Y_center, 2)) / n

    # T2 is the term involving m_hat^2:
    T2 = 0
    for j1 in range(d):
        for j2 in range(d):
            ints0 = np.zeros(M)
            for l in range(M):
                integrand = m_hat[:, j1] * p_hat_tab2[:, l, j1, j2]
                ints0[l] += integrate_0_1(integrand, dx)
            T2 += integrate_0_1(m_hat[:, j2] * ints0, dx)

    # T3 is the cross term
    T3 = 0
    for j in range(d):
        integrand_cross_term = f_hat_tab[:, j] * p_hat_tab[:, j] * m_hat[:, j]
        T3 += integrate_0_1(integrand_cross_term, dx)

    first_term = np.log(0.5 * (T1 + T2 - 2 * T3))

    # second term:
    eps = 0.001
    m_hat_col_sums = np.sum(np.power(m_hat, 2), axis=0)
    valid_indices = m_hat_col_sums > eps  # Boolean array for valid columns
    second_term = np.sum(np.log(n * h[valid_indices]) / (n * h[valid_indices]))

    return first_term + second_term


def integrate2d(array, arrLength, dx):
    arrSum = 4 * np.sum(array)
    for l in range(arrLength):
        arrSum -= 2 * array[0, l] + 2 * array[-1, l] + 2 * array[l, 0] + 2 * array[l, -1]
    return arrSum * dx ** 2 / 4


def BIC_estimate2(mHat, betaHat, gridData):
    # First term: log( 0.5 * (T1 + T2 - 2 * T3) )

    # T1 is the Y- x^T@Beta term variance term
    mean_const = np.mean(gridData.Y_center - gridData.stdX @ betaHat)
    T1 = np.sum(np.power(gridData.Y - gridData.stdX @ betaHat - mean_const, 2)) / gridData.n

    # T2 is the term involving m_hat^2:
    # Diagonal terms
    T2 = 0
    for j in range(gridData.d_c):
        T2 += integrate_0_1(np.power(mHat[:, j], 2) * gridData.pHat[:, j], gridData.dx)

    # Cross terms
    for j1 in range(gridData.d_c):
        for j2 in range(j1 + 1, gridData.d_c):
            integrand = np.outer(mHat[:, j1], mHat[:, j2]) * gridData.pHat2[:, :, j1, j2]
            T2 += 2 * integrate2d(integrand, gridData.M, gridData.dx)

    # T3 is the cross term
    T3 = 0
    marginalNW = gridData.fHat - np.sum(gridData.X_NW_table * betaHat.reshape(1, 1, gridData.d_d), axis=2)
    for j in range(gridData.d_c):
        integrand_cross_term = mHat[:, j] * marginalNW[:, j] * gridData.pHat[:, j]
        T3 += integrate_0_1(integrand_cross_term, gridData.dx)

    first_term = np.log(0.5 * (T1 + T2 - 2 * T3))

    # second term:
    eps = 0.0001
    m_hat_col_sums = np.sum(np.power(mHat, 2), axis=0)
    valid_indices = m_hat_col_sums > eps  # Boolean array for valid columns
    second_term = np.sum(np.log(gridData.n * gridData.h[valid_indices]) / (gridData.n * gridData.h[valid_indices]))
    third_term = np.log(gridData.n / gridData.n) * np.sum(np.abs(betaHat) > eps)
    return first_term + second_term + third_term


def predictorPL(mHat, betaHat, gridData):
    fin_z_grid = np.linspace(gridData.Z_min, gridData.Z_max, gridData.M)
    additive_function = AdditiveFunction(fin_z_grid, mHat, 0)
    return additive_function.predict(gridData.Z) + gridData.stdX @ betaHat + gridData.Y.mean()


def log_lik_BIC(mHat, betaHat, gridData):
    eps = 0.0001
    n = gridData.n
    ### For the penalty term...
    if gridData.d_c > 0:
        m_hat_col_sums = np.sum(np.power(mHat, 2), axis=0)
        valid_indices = m_hat_col_sums > eps  # Boolean array for valid columns
        Z_pen = np.sum(np.log(gridData.n * gridData.h[valid_indices]) / (gridData.n * gridData.h[valid_indices]))
        # Z_pen = np.sum(np.log(gridData.n*gridData.h)/(gridData.n*gridData.h))
    else:
        Z_pen = 0

    fitted = predictorPL(mHat, betaHat, gridData)

    # pen_term = (X_part.shape[1] + Z_part.shape[1])*np.log(n)/n
    pen_term = np.sum(np.abs(betaHat) > eps) * np.log(n) / n + Z_pen

    if gridData.Ytype == b"cont":
        return -0.5 * np.log(np.mean(np.power(gridData.Y - fitted, 2))) - pen_term

    if gridData.Ytype == b"disc":
        # Make sure all the fitted values are between epsilon and 1-epsilon
        eps = 0.001
        fitted_clipped = np.clip(fitted, 0 + eps, 1 - eps)
        return np.mean(gridData.Y * np.log(fitted_clipped / (1 - fitted_clipped)) +
                       np.log(1 - fitted_clipped)) - pen_term
    else:
        print("Check type")
        return None


def BIC_estimate_OG(Y, beta_hat, stdX):
    k = beta_hat.shape[0]
    n = Y.shape[0]
    return np.log(0.5 * np.mean(np.power(Y - Y.mean() - stdX @ beta_hat, 2))) + k * np.log(n) / n


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


class GridData:
    def __init__(self, Y, X):
        M, maxIter, d, Y_mean, Y_center, dx = initialize_SBF(X, Y)

        # Initialize, transform x-point grid:
        x_grid, X_center, X_min, X_max, n = initialize_x(X, M)

        # Initialize bandwidth h if h is not provided
        h = initialize_h(X_center)

        # Precompute tables
        self.KhTab = Kh_table_generator(X_center, x_grid, h)
        self.pHat = phat_table_generator(X_center, x_grid, self.KhTab)
        self.pHat2 = phat2_table_generator(X_center, x_grid, self.KhTab)
        self.fHat = fhat_table_generator(Y_center, X_center, x_grid, self.pHat, self.KhTab)
        self.dx = dx
        self.M = M


class GridDataMixed:
    def __init__(self, Y, X, Z, Ytype=None, h=None):
        M, maxIter, d, Y_mean, Y_center, dx = initialize_SBF(X, Y)

        # Initialize, transform x-point grid:
        z_grid, Z_center, Z_min, Z_max, n = initialize_x(Z, M)

        # Initialize bandwidth h if h is not provided
        if h is None:
            h = initialize_h(Z_center)

        # Precompute tables
        self.KhTab = Kh_table_generator(Z_center, z_grid, h)
        self.pHat = phat_table_generator(Z_center, z_grid, self.KhTab)
        self.pHat2 = phat2_table_generator(Z_center, z_grid, self.KhTab)
        self.fHat = fhat_table_generator(Y_center, Z_center, z_grid, self.pHat, self.KhTab)
        self.dx = dx
        self.M = M
        self.d_d = X.shape[1]
        self.d_c = Z.shape[1]
        self.Y = Y
        self.Y_center = Y_center
        self.X = X
        self.Z = Z
        self.n = n
        self.stdX = standardize(self.X)
        self.maxIter = maxIter
        self.h = h
        self.Z_min = Z_min
        self.Z_max = Z_max
        self.Ytype = Ytype

        X_NW_table = np.zeros((M, self.d_c, self.d_d))
        for k in range(self.d_d):
            X_NW_table[:, :, k] = NW_estimator(Y=self.stdX[:, k], X=Z, phat_table=self.pHat, Kh_table=self.KhTab)
        self.X_NW_table = X_NW_table


"""
X1 = np.random.normal(size=(10,3))
Y = np.random.normal(size=(10,1))

obj = SBF(X1,Y,1,"LL")
obj.evaluate(2)



MM = 400
np.random.seed(10)
data1 = NASSM.SyntheticData.SynthNonLinear(n=100, latentDim=2, obsDim=6, sigmay=2, sigmax=0.1, indep=0)
xpp = initialize_x(data1.X,M=MM)
KhTab = Kh_table_generator(data1.X,xpp,h=0.5)
KhTab
KhTab2 = Kh_table_generator2(data1.X,xpp,h=0.5)
np.sum(np.power(KhTab-KhTab2,2))
et = xpp
to = data1.X
et-to


phats = phat_table_generator(data1.X,xpp,KhTab)

fhat_table_generator(data1.Y[:,1], data1.X, xpp, phats, KhTab )-\
fhat_table_generator2(data1.Y[:,1], data1.X, xpp, phats, KhTab )

# Check to see how often it fails...

    data1 = NASSM.SyntheticData.SynthNonLinear(n=100, latentDim=2, obsDim=6, sigmay=2, sigmax=0.1, indep=0)

    XX = data1.X
    YY = data1.Y
    YY = YY - np.mean(YY, axis=0)
    hh = 0.1
    MM = 200

    print(i)
    xpp = initialize_x(data1.X, M=MM)
    KhTab = Kh_table_generator(data1.X, xpp, h=0.1)
    phats = phat_table_generator(data1.X, xpp, KhTab)
    #import matplotlib.pyplot as plt
    print("phats min",np.min(phats))
    #fig = plt.figure(figsize=(12, 12))
    #plt.plot(range(MM), phats[:, 0])
    #plt.show()
    #final = SBF(X=XX, Y=YY, M=200, h=0.1, maxIter=2000)

xpp = initialize_x(data1.X,M=MM)
KhTab = Kh_table_generator(data1.X,xpp,h=0.5)
phats = phat_table_generator(data1.X,xpp,KhTab)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
plt.plot(range(MM),phats[:,0])
plt.show()

data1 = NASSM.SyntheticData.SynthNonLinear(n=100, latentDim=2, obsDim=6, sigmay=2, sigmax=0.1, indep=0)


XX = data1.X
YY = data1.Y
YY = YY - np.mean(YY, axis=0)
hh = 0.1
MM = 200

final = SBF(X=XX, Y=YY, M=200, h=0.1, maxIter=2000)


xpp = initialize_x(data1.X,M=MM)
KhTab = Kh_table_generator(data1.X,xpp,h=0.5)
phats = phat_table_generator(data1.X,xpp,KhTab)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
plt.plot(range(MM),phats[:,0])
plt.show()

# Now we plot the functions
dim = 5
from matplotlib import cm


fig = plt.figure(figsize=(12, 12))
ax = plt.axes()
x0 = data1.X[:, 0]
x1 = data1.X[:, 1]
plt.scatter(x0,x1)
plt.
def plotter2d(final):
    M = final.x_points.shape[0]
    output = np.zeros((M, M))
    meshex = np.zeros((M, M))
    meshey = np.zeros((M, M))
    for l in range(M):
        for k in range(M):
            #output[l,k] = final.m_points[l,0,dim] + final.m_points[k,1,dim]
            meshex[l, k] = final.x_points[l, 0]
            meshey[l, k] = final.x_points[k, 1]
            output[l, k] = final.evaluate(np.array([meshex[l, k], meshey[l, k]]))[dim]
    return output, meshex, meshey


meshed, X1, X2 = plotter2d(final)

# X1,X2 = np.meshgrid(xpoint[:,0],(xpoint[:,1]))
fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, meshed, cmap=cm.coolwarm, alpha=0.3, cstride=10, rstride=10, edgecolors='k', lw=0.2)
x0 = data1.X[:, 0]
x1 = data1.X[:, 1]
#yy = data1.Y[:, 0]
ax.scatter(x0, x1, YY[:,dim])
plt.show()

def fhat_table_generator(Y, X, x_points, phat_table, Kh_table):
    # Output is (M,d)
    M = x_points.shape[0]
    d = x_points.shape[1]
    n = X.shape[0]
    output = np.zeros((M, d))
    for j in range(d):
        for l in range(M):
            sumvar = 0
            for i in range(n):
                sumvar += Y[i] * Kh_table[l, i, j]
            # print(sumvar*(1/(phat_table[l,j])))
            output[l, j] = (1 / n) * sumvar * (1 / (phat_table[l, j]))
    return output

def Kh_table_generator(X, x_points, h):
    # output: (M,n,d)
    # First we plug in unnormalized kernel estimates
    d = X.shape[1]
    n = X.shape[0]
    M = x_points.shape[0]
    output = np.zeros((M, n, d))
    for j in range(d):
        for k in range(n):
            for i in range(M):
                output[i, k, j] = ker(x_points[i, j] - X[k, j], h)
    # Normalization: for each j in range(d) and each k in range(n),
    for j in range(d):
        for k in range(n):
            normconst = np.trapz(output[:, k, j], x_points[:, j])
            output[:, k, j] = output[:, k, j] / normconst
    return output
"""
