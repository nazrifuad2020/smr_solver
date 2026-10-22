# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:55:40 2026

@author: nazmo
"""

import numpy as np
from scipy.special import roots_jacobi

def reactor_radial_grid(N):
    """Jacobi P_N^(1,0) on [0,1], cylindrical weight x."""
    roots, _ = roots_jacobi(N, 0, 1)     # (N, 1, 0)     # on [-1,1]
    x_int = np.sort((1 + roots) / 2)
    x = np.concatenate(([0.0], x_int, [1.0]))
    A = _lagrange_first_derivative(x)
    B = A @ A
    M = A + np.diag(x) @ B               # cylindrical Laplacian
    return x, A, B, M

def pellet_radial_grid(N):
    """Jacobi P_N^(2,0) on [0,1], spherical weight x^2."""
    roots, _ = roots_jacobi(N, 0, 2)  # (N, 2, 0)
    x_int = np.sort((1 + roots) / 2)
    x = np.concatenate(([0.0], x_int, [1.0]))
    A = _lagrange_first_derivative(x)
    B = A @ A
    # Spherical operator M = B + (2/x) A, with center limit M_0 = 3B_0
    M = np.zeros_like(B)
    for j in range(len(x)):
        if x[j] > 1e-14:
            M[j, :] = B[j, :] + (2.0 / x[j]) * A[j, :]
        else:
            M[j, :] = 3.0 * B[j, :]
    return x, A, B, M

def _lagrange_first_derivative(x):
    """First-derivative matrix for Lagrange basis on nodes x."""
    n = len(x)
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if i != j:
                prod = 1.0
                for k in range(n):
                    if k != i and k != j:
                        prod *= (x[j] - x[k]) / (x[i] - x[k])
                A[j, i] = prod / (x[i] - x[j])
            else:
                A[j, i] = sum(1.0 / (x[j] - x[k])
                              for k in range(n) if k != j)
    return A

def axial_grid(L, Nz):
    z = np.linspace(0.0, L, Nz + 1)
    dz = L / Nz
    return z, dz

# Quadrature weights for the pellet (weight x^2 on [0,1])
def pellet_quadrature_weights(N):
    # Use Jacobi-Gauss weights via roots_jacobi
    roots, w = roots_jacobi(N, 0, 2)  # (N, 2, 0)
    x_int = np.sort((1 + roots) / 2)
    w = w / 8.0                     # scale from [-1,1] to [0,1]
    return x_int, w               # interior points and weights