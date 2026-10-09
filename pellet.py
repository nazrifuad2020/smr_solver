# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:56:58 2026

@author: nazmo
"""

import numpy as np
from config import (D_e_CH4, D_e_H2O, D_e_CO, D_e_CO2, D_e_H2,
                    k_e_s, rho_p, R_p, R_SI)
from kinetics import intrinsic_rates
from grid import pellet_radial_grid, pellet_quadrature_weights
from deactivation import coke_formation_rate

SPECIES = ['CH4', 'H2O', 'CO', 'CO2', 'H2']
D_e = {'CH4': D_e_CH4, 'H2O': D_e_H2O, 'CO': D_e_CO,
       'CO2': D_e_CO2, 'H2': D_e_H2}
NU = {
    'SMR': {'CH4': -1, 'H2O': -1, 'CO': +1, 'CO2': 0, 'H2': +3},
    'WGS': {'CH4':  0, 'H2O': -1, 'CO': -1, 'CO2': +1, 'H2': +1},
    'GRR': {'CH4': -1, 'H2O': -2, 'CO':  0, 'CO2': +1, 'H2': +4},
}
dH = {'SMR': 206.1, 'WGS': -41.15, 'GRR': 165.0}

_xp, _Ap, _Bp, _Mp = pellet_radial_grid(10)   # N_p = 10

def solve_pellet(C_bulk, T_bulk, a, k_c, h_f, P_total):
    """
    Solve steady-state pellet balances for surface concentrations and T.
    Returns: C_surf dict, T_surf, effectiveness factors.
    Uses Newton iteration on interior nodes.
    """
    N = len(_xp) - 2
    # Flatten unknowns: [C_CH4..., C_H2O..., ..., T] at interior nodes
    n_sp = len(SPECIES)
    n_unk = n_sp * N + N
    y = np.zeros(n_unk)
    # Initial guess: bulk values
    for i, sp in enumerate(SPECIES):
        y[i*N:(i+1)*N] = C_bulk[sp]
    y[-N:] = T_bulk

    def residual(y):
        C = {sp: np.zeros(N+2) for sp in SPECIES}
        for i, sp in enumerate(SPECIES):
            C[sp][1:-1] = y[i*N:(i+1)*N]
            C[sp][-1] = C_bulk[sp]         # surface = bulk (first iteration)
            C[sp][0]  = y[i*N]             # center: symmetric
        T = np.zeros(N+2)
        T[1:-1] = y[-N:]
        T[-1] = T_bulk
        T[0]  = y[-N]

        res = np.zeros(n_unk)
        for j in range(1, N+1):             # interior nodes
            # Partial pressures [bar]
            p = {sp: C[sp][j] * R_SI * T[j] / 1e5 for sp in SPECIES}
            r1, r2, r3 = intrinsic_rates(T[j], p)
            a_eff = a
            rxn = {'SMR': a_eff*r1, 'WGS': a_eff*r2, 'GRR': a_eff*r3}
            for i, sp in enumerate(SPECIES):
                src = rho_p * sum(NU[r][sp]*rxn[r] for r in rxn)  # mol/m^3/s
                # Convert kmol/kg/h -> mol/m^3/s: ×1000/3600
                src *= 1000.0/3600.0
                diff = (D_e[sp] / R_p**2) * sum(_Mp[j, k]*C[sp][k]
                                                for k in range(N+2))
                res[i*N + (j-1)] = diff + src
            # Energy
            q_src = rho_p * sum(dH[r]*rxn[r]*1000.0/3600.0 for r in rxn)
            diff_T = (k_e_s / R_p**2) * sum(_Mp[j, k]*T[k] for k in range(N+2))
            res[n_sp*N + (j-1)] = diff_T + q_src
        return res

    from scipy.optimize import root
    sol = root(residual, y, method='hybr', options={'xtol': 1e-10})
    # Unpack
    C_surf = {}
    for i, sp in enumerate(SPECIES):
        C_surf[sp] = sol.x[i*N + N - 1]     # last interior node
    T_surf = sol.x[-1]
    return C_surf, T_surf

def effectiveness_factor(T, C, a):
    """Compute η for the SMR reaction via quadrature."""
    x_int, w = pellet_quadrature_weights(len(_xp)-2)
    r_vals = np.zeros(len(x_int))
    for i, x in enumerate(x_int):
        p = {sp: C[sp]*R_SI*T/1e5 for sp in SPECIES}
        r1, _, _ = intrinsic_rates(T, p)
        r_vals[i] = a * r1
    # Surface rate
    p_s = {sp: C[sp]*R_SI*T/1e5 for sp in SPECIES}
    r1_s, _, _ = intrinsic_rates(T, p_s)
    return 3.0 * np.sum(w * r_vals * x_int**2) / (r1_s + 1e-30)