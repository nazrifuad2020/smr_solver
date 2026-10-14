# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:57:27 2026

@author: nazmo
"""

import numpy as np
from scipy.optimize import root

from config import (N_r, N_p, N_SPECIES, CP_MOL, D_e, eps_bed, rho_p, D_er, 
                    k_er_f, R_t, R_p, u_s, rho_b, h_f, h_w, k_c, k_e_s, a_v, 
                    R_SI, SPECIES)
from config import reaction_rates, species_source_terms, heat_source_term
from grid import reactor_radial_grid, pellet_radial_grid
from pellet import compute_effectiveness_factors


# ============================================================
# Precompute collocation data (once at import)
# ============================================================
_xr, _Ar, _Br, _Mr = reactor_radial_grid(N_r)   # cylindrical
_xp, _Ap, _Bp, _Mp = pellet_radial_grid(N_p)    # spherical

_N_K   = len(_xr)                                # N_r + 2 radial points
_N_P   = len(_xp)                                # N_p + 2 pellet points
_N_SP  = N_SPECIES
_BLOCK = (_N_SP + 1) * (_N_P + 1)                # unknowns per radial point


# ============================================================
# Packing / unpacking utilities
# ============================================================
def _initial_guess_hetero(C_in, T_in):
    """Uniform initial guess: fluid and pellet equal to inlet conditions."""
    y0 = np.zeros(_N_K * _BLOCK)
    for k in range(_N_K):
        base = k * _BLOCK
        for i, sp in enumerate(SPECIES):
            y0[base + i] = C_in[sp]
        y0[base + _N_SP] = T_in
        for p in range(_N_P):
            poff = base + _N_SP + 1 + p * _N_SP
            for i, sp in enumerate(SPECIES):
                y0[poff + i] = C_in[sp]
        toff = base + _N_SP + 1 + _N_SP * _N_P
        for p in range(_N_P):
            y0[toff + p] = T_in
    return y0


def _initial_guess_homo(C_in, T_in):
    n_unk = len(SPECIES)*_N_K + _N_K
    y0 = np.zeros(n_unk)
    for i, sp in enumerate(SPECIES):
        y0[i*_N_K:(i+1)*_N_K] = C_in[sp]
    y0[-_N_K:] = T_in
    return y0


def _unpack_hetero(y, j, C_bulk, T_bulk, C_pellet, T_pellet):
    """Store the solution at axial node j into the global arrays."""
    for k in range(_N_K):
        base = k * _BLOCK
        for i, sp in enumerate(SPECIES):
            C_bulk[sp][j, k] = y[base + i]
        T_bulk[j, k] = y[base + _N_SP]
        for p in range(_N_P):
            poff = base + _N_SP + 1 + p * _N_SP
            for i, sp in enumerate(SPECIES):
                C_pellet[sp][j, k, p] = y[poff + i]
        toff = base + _N_SP + 1 + _N_SP * _N_P
        for p in range(_N_P):
            T_pellet[j, k, p] = y[toff + p]
            
            
def _unpack_homo(y, j, C_bulk, T_bulk):
    for i, sp in enumerate(SPECIES):
        C_bulk[sp][j, :_N_K] = y[i*_N_K:(i+1)*_N_K]
    T_bulk[j, :_N_K] = y[-_N_K:]


# ============================================================
# Residual assembly for one axial node
# ============================================================
def _residual_hetero(y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node):
    """
    Residual of the coupled fluid + pellet system at one axial node.

    Layout of y per radial point k (block size = (N_SP+1)*(N_P+1)):
        C_b[k, 0..N_SP-1]           -> fluid bulk concentrations
        T_b[k]                      -> fluid bulk temperature
        C_s[k, i, p] for i, p       -> pellet concentrations (all points)
        T_s[k, p] for p             -> pellet temperatures (all points)
    """
    res = np.zeros_like(y)

    # ---- Extract structured views ----
    C_b = np.zeros((_N_K, _N_SP))
    T_b = np.zeros(_N_K)
    C_s = np.zeros((_N_K, _N_SP, _N_P))
    T_s = np.zeros((_N_K, _N_P))

    for k in range(_N_K):
        base = k * _BLOCK
        C_b[k, :] = y[base : base + _N_SP]
        T_b[k]    = y[base + _N_SP]
        for p in range(_N_P):
            poff = base + _N_SP + 1 + p * _N_SP
            C_s[k, :, p] = y[poff : poff + _N_SP]
        toff = base + _N_SP + 1 + _N_SP * _N_P
        T_s[k, :] = y[toff : toff + _N_P]

    # ============================================================
    # Fluid-phase equations
    # ============================================================
    for k in range(_N_K):
        base = k * _BLOCK

        if k < _N_K - 1:
            # ---------- Interior radial points (incl. center) ----------
            # Mass balance for each species
            for i in range(_N_SP):
                rad = sum(_Mr[k, m] * C_b[m, i] for m in range(_N_K))
                rad *= eps_bed * D_er * 4.0 / R_t**2

                inter = -k_c * a_v * (C_b[k, i] - C_s[k, i, _N_P - 1])

                conv = -u_s * (C_b[k, i] - C_prev[SPECIES[i]][k]) / dz

                res[base + i] = rad + inter + conv

            # Energy balance
            rad_T = sum(_Mr[k, m] * T_b[m] for m in range(_N_K))
            rad_T *= k_er_f * 4.0 / R_t**2

            inter_T = -h_f * a_v * (T_b[k] - T_s[k, _N_P - 1])

            sum_C_cp = sum(C_b[k, i] * CP_MOL[SPECIES[i]]
                           for i in range(_N_SP))
            conv_T = -u_s * sum_C_cp * (T_b[k] - T_prev[k]) / dz

            res[base + _N_SP] = rad_T + inter_T + conv_T

        else:
            # ---------- Wall point (k = N_K - 1) ----------
            # Mass: zero flux at wall -> dC/dx = 0
            for i in range(_N_SP):
                dCdx = sum(_Ar[_N_K - 1, m] * C_b[m, i]
                           for m in range(_N_K))
                res[base + i] = dCdx

            # Energy: (2 k_er / R_t) dT/dx|_wall = h_w (T_wall - T_b[wall])
            dTdx = sum(_Ar[_N_K - 1, m] * T_b[m] for m in range(_N_K))
            res[base + _N_SP] = (
                (2.0 * k_er_f / R_t) * dTdx
                - h_w * (T_wall - T_b[_N_K - 1])
            )

    # ============================================================
    # Pellet equations (for every radial point)
    # ============================================================
    for k in range(_N_K):
        base = k * _BLOCK
        a_k = a_node[k]

        for p in range(_N_P):
            poff = base + _N_SP + 1 + p * _N_SP
            toff = base + _N_SP + 1 + _N_SP * _N_P + p

            if p < _N_P - 1:
                # ---------- Center and interior pellet points: PDE ----------
                T_loc = T_s[k, p]
                p_part = {sp: C_s[k, i, p] * R_SI * T_loc / 1e5
                          for i, sp in enumerate(SPECIES)}
                rxn = reaction_rates(T_loc, p_part, a_k)
                S = species_source_terms(rxn)
                for sp in S:
                    S[sp] *= rho_p * 1000.0 / 3600.0    # mol/(m^3·s)
                Q = heat_source_term(rxn)
                Q *= rho_p * 1000.0 / 3600.0 * 1000.0   # W/m^3

                # r1, r2, r3 = intrinsic_rates(T_loc, p_part)
                # rxn = np.array([a_k * r1, a_k * r2, a_k * r3])
                # S = NU @ rxn                                    # kmol/(kg_cat·h)
                # S_si = S * rho_p * 1000.0 / 3600.0              # mol/(m^3·s)
                # Q = (-DELTA_H_vec @ rxn) * rho_p * 1000.0 / 3600.0 * 1000.0  # W/m^3

                # Mass balance
                for i in range(_N_SP):
                    diff = sum(_Mp[p, m] * C_s[k, i, m]
                               for m in range(_N_P))
                    diff *= D_e[SPECIES[i]] / R_p**2
                    res[poff + i] = diff + S[SPECIES[i]]

                # Energy balance
                diff_T = sum(_Mp[p, m] * T_s[k, m] for m in range(_N_P))
                diff_T *= k_e_s / R_p**2
                res[toff] = diff_T + Q

            else:
                # ---------- Pellet surface: flux-matching BC ----------
                for i in range(_N_SP):
                    dCdx = sum(_Ap[_N_P - 1, m] * C_s[k, i, m]
                               for m in range(_N_P))
                    flux_int = (D_e[SPECIES[i]] / R_p) * dCdx
                    flux_ext = k_c * (C_b[k, i] - C_s[k, i, _N_P - 1])
                    res[poff + i] = flux_int - flux_ext

                dTdx = sum(_Ap[_N_P - 1, m] * T_s[k, m]
                           for m in range(_N_P))
                flux_T_int = (k_e_s / R_p) * dTdx
                flux_T_ext = h_f * (T_b[k] - T_s[k, _N_P - 1])
                res[toff] = flux_T_int - flux_T_ext

    return res


def _residual_homo(y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node):
    S = {sp: 0.0 for sp in SPECIES}
    Q = 0.0
    Cj = {sp: np.zeros(_N_K) for sp in SPECIES}
    for i, sp in enumerate(SPECIES):
        Cj[sp][:_N_K] = y[i*_N_K:(i+1)*_N_K]
    Tj = np.zeros(_N_K)
    Tj[:_N_K] = y[-_N_K:]
    
    n_unk = len(SPECIES)*_N_K + _N_K

    res = np.zeros(n_unk)
    for k in range(_N_K):
        a_loc = a_node[k]
        p = {sp: Cj[sp][k]*R_SI*Tj[k]/1e5 for sp in SPECIES}
        
        if k < _N_K - 1:
            rxn = reaction_rates(Tj[k], p, a_loc)
            S = species_source_terms(rxn)
            for sp in S:
                S[sp] *= rho_p * 1000.0 / 3600.0    # mol/(m^3·s)
            Q = heat_source_term(rxn)
            Q *= rho_p * 1000.0 / 3600.0 * 1000.0   # W/m^3
            for i, sp in enumerate(SPECIES):
                # Radial diffusion
                rad = (D_er * 4.0/R_t**2) * sum(_Mr[k, m]*Cj[sp][m]
                                                for m in range(_N_K))
                # Axial convection (backward difference for stability)
                conv = -u_s * (Cj[sp][k] - C_prev[sp][k]) / dz
                res[i*_N_K+k] = rad + conv + S[sp]
            # Energy
            rad_T = (k_er_f * 4.0/R_t**2) * sum(_Mr[k, m]*Tj[m]
                                            for m in range(_N_K))
            conv_T = -u_s * rho_b * 2.5 * (Tj[k] - T_prev[k]) / dz
            res[-_N_K+k] = rad_T + conv_T + Q
            
        else:
            # ---------- Wall point (k = N_K - 1) ----------
            # Mass: zero flux at wall -> dC/dx = 0
            for i, sp in enumerate(SPECIES):
                res[i*_N_K+k] = sum(_Ar[_N_K - 1, m] * Cj[sp][m]
                                    for m in range(_N_K))
            # Energy: (2 k_er / R_t) dT/dx|_wall = h_w (T_wall - T_b[wall])
            dTdx = sum(_Ar[_N_K - 1, m] * Tj[m] for m in range(_N_K))
            res[-1] = ((2.0 * k_er_f / R_t) * dTdx - h_w * (T_wall - Tj[_N_K - 1]))
        
    return res


def solve_reactor(
    z, 
    dz, 
    activity_profile, 
    C_in, 
    T_in, 
    T_wall,
    tol=1e-8, 
    maxfev=10000, 
    verbose=True,
    heterogeneous=True,
):
    """
    March axially through the reactor, solving the coupled fluid + pellet
    system simultaneously at each axial node.

    Parameters
    ----------
    z : ndarray, shape (Nz+1,)
        Axial grid points [m].
    dz : float
        Axial step size [m].
    activity_profile : ndarray, shape (Nz+1, N_K)
        Catalyst activity at each axial and radial location [-].
    C_in : dict
        Inlet concentrations [mol/m^3] keyed by SPECIES names.
    T_in : float
        Inlet temperature [K].
    T_wall : float
        Wall temperature [K].
    tol : float
        Newton tolerance.
    maxfev : int
        Maximum number of function evaluations per node.
    verbose : bool
        Print progress.

    Returns
    -------
    C_bulk   : dict of ndarray (Nz+1, N_K)   fluid bulk concentrations
    T_bulk   : ndarray (Nz+1, N_K)           fluid bulk temperatures
    C_pellet : dict of ndarray (Nz+1, N_K, N_P)  pellet concentrations
    T_pellet : ndarray (Nz+1, N_K, N_P)      pellet temperatures
    """
    Nz = len(z) - 1
    
    C_bulk   = {sp: np.zeros((Nz + 1, _N_K)) for sp in SPECIES}
    T_bulk   = np.zeros((Nz + 1, _N_K))
    C_pellet = {sp: np.zeros((Nz + 1, _N_K, _N_P)) for sp in SPECIES}
    T_pellet = np.zeros((Nz + 1, _N_K, _N_P))
    
    # ---- Inlet conditions ----
    for sp in SPECIES:
        C_bulk[sp][0, :] = C_in[sp]
        C_pellet[sp][0, :, :] = C_in[sp]
    T_bulk[0, :] = T_in
    T_pellet[0, :, :] = T_in
    
    # ---- Initial guess ----
    if heterogeneous:
        y_guess = _initial_guess_hetero(C_in, T_in)
    else:
        y_guess = _initial_guess_homo(C_in, T_in)
        
    for j in range(1, Nz + 1):
        C_prev = {sp: C_bulk[sp][j - 1, :] for sp in SPECIES}
        T_prev = T_bulk[j - 1, :].copy()
        a_node = activity_profile[j, :]
        
        def residual(y):
            if heterogeneous:
                return _residual_hetero(
                    y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node
                )
            else:
                return _residual_homo(
                    y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node
                )
        
        sol = root(
            residual, 
            y_guess, 
            method='hybr', 
            options={'xtol': tol, 'maxfev': maxfev},
        )
        
        y_guess = sol.x.copy()
        if heterogeneous:
            _unpack_hetero(sol.x, j, C_bulk, T_bulk, C_pellet, T_pellet)
        else:
            _unpack_homo(sol.x, j, C_bulk, T_bulk)
    
    eta = compute_effectiveness_factors(
        C_bulk, T_bulk, C_pellet, T_pellet, activity_profile
    )

    return C_bulk, T_bulk, C_pellet, T_pellet, eta