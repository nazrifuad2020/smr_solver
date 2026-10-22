# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:56:58 2026

@author: nazmo
"""

import numpy as np
from config import SPECIES, NU, D_e, k_e_s, rho_p, R_p, R_SI, DELTA_H as dH
from kinetics import intrinsic_rates
from grid import pellet_radial_grid, pellet_quadrature_weights


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

# ============================================================
# Convenience: effectiveness factor from the full pellet solution
# ============================================================
def compute_effectiveness_factors(C_bulk, T_bulk, C_pellet, T_pellet, 
                                  activity_profile):
    """
    Compute the SMR effectiveness factor at every (z, r) location
    by integrating the intrinsic rate over the pellet volume.
    """
    Nz = C_bulk[SPECIES[0]].shape[0] - 1
    # Nk = C_bulk[SPECIES[0]].shape[1]
    Np = C_pellet[SPECIES[0]].shape[1]
    
    N_p = Np - 2
    _, w_pel = pellet_quadrature_weights(N_p)
    
    eta = np.zeros(Nz + 1)

    for j in range(Nz + 1):
        a_j = activity_profile[j]
        # Volume-averaged rate
        r_avg = 0.0
        for idx, p in enumerate(range(1, Np - 1)):
            T_loc = T_pellet[j, p]
            p_part = {sp: C_pellet[sp][j, p] * R_SI * T_loc / 1e5 
                      for sp in SPECIES}
            r1, _, _ = intrinsic_rates(T_loc, p_part)
            r_avg += w_pel[idx] * a_j * r1 # * _xp[p]**2  
        r_avg *= 3.0

        # Surface rate
        T_surf = T_pellet[j, Np - 1]
        p_surf = {sp: C_pellet[sp][j, Np - 1] * R_SI * T_surf / 1e5
                  for sp in SPECIES}
        r1_s, _, _ = intrinsic_rates(T_surf, p_surf)
        eta[j] = r_avg / (a_j * r1_s + 1e-30)

    return eta

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