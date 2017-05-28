#!/usr/bin/env python
# -*- coding: utf8 -*-
# Author: Zhang Sheng
# Time: 2017/05/28 10:40

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    """RBF PCA implementation.

    Parameters
    ----------
    X: {Numpy ndarray}, shape = [n_samples, n_features]
    gamma: float
        Tuning parameter of the RBF kernel
    n_components: list
        Number of principal components to return

    Returns
    -------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset

    lambdas: list
        Eigenvalues
    """

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')

    # convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # obtaining eigenpairs from  the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    lambdas = [eigvals[-i] for i in range(1, n_components+1)]
    return X_pc, lambdas

