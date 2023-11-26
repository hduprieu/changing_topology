import numpy as np
import numpy.random as rd
import scipy.optimize as optim
import matplotlib.pyplot as plt

def DSGD_toypbm(lr, d, n, W, T):
    """
    D-SGD algorithm for the toy-problem defined in section 3 (minimization of 1/2*||x||^2)

    Inputs :
    ------------
    lr: learning rate
    d: dimension
    n: number of agents
    W: gossip matrix (symmetric, stochastic, of size n*n)
    T: number of time steps

    Prints at each iteration the state x obtained by worker i = 0
    """

    x0 = rd.normal(0,1, size = (d))
    X = np.block([[x0]]*n)

    for _ in range(T):
        for i in range(n):
            v = rd.normal(0,1, size = (d,1))
            X[i,:] = X[i,:] - lr * v.dot(v.T).dot(X[i,:])

        X = W.dot(X)

        print(X[0,:])

n = 10
d = 4

# Ring topology gossip matrix

W = np.block([[1/5 * np.ones((5,5)),np.zeros((5,5))],[np.zeros((5,5)),1/5 * np.ones((5,5))]])
W[4,4] = 1/10
W[4,5] = 1/10
W[5,4] = 1/10
W[5,5] = 1/10

DSGD_toypbm(0.4, d, n, W, 50)

def nW(gamma, W):
    """
    Computes the effective number of neighbours of the gossip matrix W for the decay parameter gamma
    """
    lambdas = np.linalg.eigvals(W)
    denominator = np.mean(lambdas**2 / (1 - gamma*lambdas**2))
    return 1/(1-gamma) * 1/denominator

# The effective number of neighbours for the alone topology (W = I) should be 1 independently of gamma
nW(.5, np.eye(10))

def rate_toypbm(lr, zeta, W):
    """
    Computes the convergence rate of D-SGD in the toy-problem case

    Inputs:
    ----------------
    lr: learning rate of D-SGD
    zeta: noise level (= d + 2 in the case of the toy-problem in dimension d)
    W: gossip matrix used for D-SGD

    Outputs:
    ----------------
    r: convergence rate of D-SGD
    """
    f_tosolve = lambda r : r - (1 - (1 - lr)**2 - (zeta - 1) * lr**2 / nW((1-lr)**2 / (1-r), W))
    r = optim.fsolve(f_tosolve)
    return r

rate_toypbm(0.5, 6, np.eye(10))