# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:57:27 2026

@author: nazmo
"""

import numpy as np
from config import (eps_bed, D_er, k_er_f, R_t, u_s, rho_b, h_f, k_c,
                    a_v, R_SI, C_total_in, P_total, y_in)
from grid import reactor_radial_grid
from kinetics import intrinsic_rates
from pellet import solve_pellet, SPECIES

_xr, _Ar, _Br, _Mr = reactor_radial_grid(6)   # N_r = 6
N_r = len(_xr) - 2

def solve_reactor(z, dz, activity_profile, C_in, T_in, T_wall):
    """
    March axially through the reactor solving the coupled
    fluid + pellet equations at each node.
    Returns: C(z,r), T(z,r), a(z,r) profiles.
    """
    Nz = len(z) - 1
    n_r = len(_xr)
    C = {sp: np.zeros((Nz+1, n_r)) for sp in SPECIES}
    T = np.zeros((Nz+1, n_r))

    # Inlet
    for sp in SPECIES:
        C[sp][0, :] = C_in[sp]
    T[0, :] = T_in

    from scipy.optimize import root

    for j in range(1, Nz+1):
        # Unknowns at node j: C_sp(r_k), T(r_k) for k=0..N_r
        # (wall values handled by BCs; interior + center solved)
        n_unk = len(SPECIES)*(N_r+1) + (N_r+1)
        y = np.zeros(n_unk)
        for i, sp in enumerate(SPECIES):
            y[i*(N_r+1):(i+1)*(N_r+1)] = C[sp][j-1, :N_r+1]
        y[-N_r-1:] = T[j-1, :N_r+1]

        def residual(y):
            Cj = {sp: np.zeros(n_r) for sp in SPECIES}
            for i, sp in enumerate(SPECIES):
                Cj[sp][:N_r+1] = y[i*(N_r+1):(i+1)*(N_r+1)]
                Cj[sp][-1] = Cj[sp][-2]    # wall: zero flux (approx)
            Tj = np.zeros(n_r)
            Tj[:N_r+1] = y[-N_r-1:]
            Tj[-1] = T_wall                 # wall temperature

            res = np.zeros(n_unk)
            for k in range(N_r+1):
                # Interfacial fluxes via pellet solve (simplified: use bulk as surface)
                p = {sp: Cj[sp][k]*R_SI*Tj[k]/1e5 for sp in SPECIES}
                r1, r2, r3 = intrinsic_rates(Tj[k], p)
                a_loc = activity_profile[j, k]
                for i, sp in enumerate(SPECIES):
                    nu = -(r1) * (1 if sp=='CH4' else 0)  # placeholder
                    # Radial diffusion
                    rad = (D_er * 4.0/R_t**2) * sum(_Mr[k, m]*Cj[sp][m]
                                                    for m in range(n_r))
                    # Interfacial mass transfer (approximate: pellet at bulk)
                    inter = -k_c * a_v * 0.0     # placeholder; full pellet in practice
                    # Axial convection (backward difference for stability)
                    conv = -u_s * (Cj[sp][k] - C[sp][j-1, k]) / dz
                    res[i*(N_r+1)+k] = rad + inter + conv + 0.0  # + reaction source
                # Energy
                rad_T = (k_er_f * 4.0/R_t**2) * sum(_Mr[k, m]*Tj[m]
                                                    for m in range(n_r))
                conv_T = -u_s * rho_b * 2.5 * (Tj[k] - T[j-1, k]) / dz
                res[-N_r-1+k] = rad_T + conv_T
            return res

        sol = root(residual, y, method='hybr')
        for i, sp in enumerate(SPECIES):
            C[sp][j, :N_r+1] = sol.x[i*(N_r+1):(i+1)*(N_r+1)]
            C[sp][j, -1] = C[sp][j, -2]
        T[j, :N_r+1] = sol.x[-N_r-1:]
        T[j, -1] = T_wall

    return C, T