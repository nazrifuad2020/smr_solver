# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:57:27 2026

@author: nazmo
"""

import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp

from config import (N_r, N_p, N_SPECIES, CP_MOL, D_e, rho_p, D_er, k_er_f, R_t, 
                    R_p, u_s, rho_b, h_f, h_w, k_c, k_e_s, a_v, R_SI, SPECIES, 
                    C_total_in, MW)
from config import reaction_rates, species_source_terms, heat_source_term
from grid import reactor_radial_grid, pellet_radial_grid
from pellet import compute_effectiveness_factors


# ============================================================
# Precompute collocation data (once at import)
# ============================================================
_xr, _Ar, _Br, _Mr = reactor_radial_grid(N_r)   # cylindrical
_xp, _Ap, _Bp, _Mp = pellet_radial_grid(N_p)    # spherical

_N_K   = len(_xr)                               # N_r + 2 radial points
_N_P   = len(_xp)                               # N_p + 2 pellet points
_N_SP  = N_SPECIES
_BLOCK = (_N_SP + 1) * 2                        # unknowns per radial point
# _BLOCK = (_N_SP + 1) * (_N_P + 1)             # unknowns per radial point


# ============================================================
# Packing / unpacking utilities
# ============================================================
def _initial_guess_hetero_2d(C_in, T_in):
    """Uniform initial guess: fluid and pellet equal to inlet conditions."""
    y0 = np.zeros(_N_K * _BLOCK)
    for k in range(_N_K):
        base = k * _BLOCK
        for i, sp in enumerate(SPECIES):
            y0[base + i] = C_in[sp]
        y0[base + _N_SP] = T_in
        poff = base + _N_SP + 1
        for i, sp in enumerate(SPECIES):
            y0[poff + i] = C_in[sp]
        toff = poff + _N_SP
        y0[toff] = T_in
    return np.log(y0)


def _initial_guess_hetero_1d(C_in, T_in):
    """Uniform initial guess: pellet equal to inlet conditions."""
    n_unk = (_N_SP + 1) * (_N_P + 1)
    y0 = np.zeros(n_unk)
    for i, sp in enumerate(SPECIES):
        y0[i] = C_in[sp]
    y0[_N_SP] = T_in
    poff = _N_SP + 1
    for i, sp in enumerate(SPECIES):
        y0[poff+i*_N_P:poff+(i+1)*_N_P] = C_in[sp]
    y0[-_N_P:] = T_in
    return np.log(y0)


def _initial_guess_homo_2d(C_in, T_in):
    n_unk = len(SPECIES)*_N_K + _N_K
    y0 = np.zeros(n_unk)
    for i, sp in enumerate(SPECIES):
        y0[i*_N_K:(i+1)*_N_K] = C_in[sp]
    y0[-_N_K:] = T_in
    return np.log(y0)


def _initial_guess_homo_1d(C_in, T_in):
    """Uniform initial guess: pellet equal to inlet conditions."""
    n_unk = (_N_SP + 1)
    y0 = np.zeros(n_unk)
    for i, sp in enumerate(SPECIES):
        y0[i] = C_in[sp]
    y0[_N_SP] = T_in
    return np.log(y0)


def _unpack_hetero_2d(yraw, j, C_bulk, T_bulk, C_pellet, T_pellet):
    """Store the solution at axial node j into the global arrays."""
    y = np.exp(yraw)
    for k in range(_N_K):
        base = k * _BLOCK
        for i, sp in enumerate(SPECIES):
            C_bulk[sp][j, k] = y[base + i]
        T_bulk[j, k] = y[base + _N_SP]
        poff = base + _N_SP + 1
        for i, sp in enumerate(SPECIES):
            C_pellet[sp][j, k] = y[poff + i]
        toff = poff + _N_SP
        T_pellet[j, k] = y[toff]
            
            
def _unpack_hetero_1d(yraw, j, C_bulk, T_bulk, C_pellet, T_pellet):
    y = np.exp(yraw)
    for i, sp in enumerate(SPECIES):
        C_bulk[sp][j] = y[i]
    T_bulk[j] = y[_N_SP]
    poff = _N_SP + 1
    for i, sp in enumerate(SPECIES):
        C_pellet[sp][j, :] = y[poff+i*_N_P:poff+(i+1)*_N_P]
    T_pellet[j, :] = y[-_N_P:]
            
            
def _unpack_homo_2d(yraw, j, C_bulk, T_bulk):
    y = np.exp(yraw)
    for i, sp in enumerate(SPECIES):
        C_bulk[sp][j, :_N_K] = y[i*_N_K:(i+1)*_N_K]
    T_bulk[j, :_N_K] = y[-_N_K:]
    
    
def _unpack_homo_1d(yraw, j, C_bulk, T_bulk):
    y = np.exp(yraw)
    for i, sp in enumerate(SPECIES):
        C_bulk[sp][j] = y[i]
    T_bulk[j] = y[_N_SP]


# ============================================================
# Residual assembly for one axial node
# ============================================================
def _residual_hetero_2d(yraw, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node):
    """
    Residual of the coupled fluid + pellet system at one axial node.

    Layout of y per radial point k (block size = (N_SP+1)*(N_P+1)):
        C_b[k, 0..N_SP-1]           -> fluid bulk concentrations
        T_b[k]                      -> fluid bulk temperature
        C_s[k, i, p] for i, p       -> pellet concentrations (all points)
        T_s[k, p] for p             -> pellet temperatures (all points)
    """
    y = np.exp(yraw)
    res = np.zeros_like(y)

    # ---- Extract structured views ----
    C_b = np.zeros((_N_K, _N_SP))
    T_b = np.zeros(_N_K)
    C_s = np.zeros((_N_K, _N_SP))
    T_s = np.zeros(_N_K)

    for k in range(_N_K):
        base = k * _BLOCK
        C_b[k, :] = y[base : base + _N_SP]
        T_b[k]    = y[base + _N_SP]
        poff = base + _N_SP + 1
        C_s[k, :] = y[poff : poff + _N_SP]
        toff = poff + _N_SP
        T_s[k] = y[toff]

    # ============================================================
    # Fluid-phase equations
    # ============================================================
    for k in range(_N_K):
        base = k * _BLOCK
        
        if k == 0:
            for i, sp in enumerate(SPECIES):
                res[base + i] = sum(_Ar[0, m] * C_b[m, i] for m in range(_N_K))
            dTdx = sum(_Ar[0, m] * T_b[m] for m in range(_N_K))
            res[base + _N_SP] = dTdx
        elif k > 0 and k < _N_K - 1:
            # ---------- Interior radial points (incl. center) ----------
            # Mass balance for each species
            for i in range(_N_SP):
                rad = sum(_Mr[k, m] * C_b[m, i] for m in range(_N_K))
                rad *= D_er * 4.0 / R_t**2

                inter = k_c * a_v * (C_s[k, i] - C_b[k, i])

                conv = u_s * (C_b[k, i] - C_prev[SPECIES[i]][k]) / dz

                res[base + i] = rad + inter - conv 

            # Energy balance
            rad_T = sum(_Mr[k, m] * T_b[m] for m in range(_N_K))
            rad_T *= k_er_f * 4.0 / R_t**2

            inter_T = h_f * a_v * (T_s[k] - T_b[k])

            sum_C_cp = sum(C_b[k, i] / C_total_in * CP_MOL[SPECIES[i]] 
                           for i in range(_N_SP))
            mix_MW = sum(C_b[k, i] / C_total_in * MW[SPECIES[i]] 
                         for i in range(_N_SP))
            cp = sum_C_cp / mix_MW
            p_part = {sp: C_b[k, i] * R_SI * T_b[k] / 1e5 
                      for i, sp in enumerate(SPECIES)}
            p_tot = sum(p_part[sp] for sp in SPECIES) # bar
            rho_g = p_tot * mix_MW / R_SI / T_b[k] * 1e5 / 1000    # kg/m^3
            conv_T = u_s * rho_g * cp * 1000 * (T_b[k] - T_prev[k]) / dz
            
            res[base + _N_SP] = rad_T + inter_T - conv_T

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
                (2.0 * k_er_f / R_t) * dTdx + h_w * (T_b[_N_K - 1] - T_wall)
            )

    # ============================================================
    # Pellet equations (for every radial point)
    # ============================================================
    for k in range(_N_K):
        base = k * _BLOCK
        a_k = a_node[k]
        
        T_loc = T_s[k]
        p_part = {sp: C_s[k, i] * R_SI * T_loc / 1e5 
                  for i, sp in enumerate(SPECIES)}
        rxn = reaction_rates(T_loc, p_part, a_k)
        S = species_source_terms(rxn)
        for sp in S:
            S[sp] *= rho_b * 1000.0 / 3600.0    # mol/(m^3·s)
        Q = heat_source_term(rxn)
        Q *= rho_b * 1000.0 / 3600.0   # W/m^3
        
        poff = base + _N_SP + 1
        for i in range(_N_SP):
            flux_ext = k_c * a_v * (C_s[k, i] - C_b[k, i])
            res[poff + i] = S[SPECIES[i]] - flux_ext
        
        toff = poff + _N_SP
        flux_T_ext = h_f * a_v * (T_s[k] - T_b[k])
        res[toff] = Q - flux_T_ext

    return res


def _residual_hetero_1d(yraw, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node):
    y = np.exp(yraw)
    S = {sp: 0.0 for sp in SPECIES}
    Q = 0.0
    C_b = {sp: 0.0 for sp in SPECIES}
    
    for i, sp in enumerate(SPECIES):
        C_b[sp] = y[i]
    T_b = y[_N_SP]
    
    poff = _N_SP + 1
    C_s = {sp: np.zeros(_N_P) for sp in SPECIES}
    for i, sp in enumerate(SPECIES):
        C_s[sp][:_N_P] = y[poff+i*_N_P:poff+(i+1)*_N_P]
    T_s = np.zeros(_N_P)
    T_s[:_N_P] = y[-_N_P:]
    
    n_unk = (_N_SP + 1) * (_N_P + 1)
    res = np.zeros(n_unk)
    
    # ============================================================
    # Fluid-phase equations
    # ============================================================
    # Mass balance for each species
    for i, sp in enumerate(SPECIES):
        inter = k_c * a_v * (C_b[sp] - C_s[sp][_N_P - 1])
        conv = u_s * (C_b[sp] - C_prev[sp]) / dz
        res[i] = conv + inter 
    # Energy balance
    inter_T = h_f * a_v * (T_s[_N_P - 1] - T_b)
    sum_C_cp = sum(C_b[SPECIES[i]] / C_total_in * CP_MOL[SPECIES[i]] 
                   for i in range(_N_SP))
    mix_MW = sum(C_b[SPECIES[i]] / C_total_in * MW[SPECIES[i]] 
                 for i in range(_N_SP))
    cp = sum_C_cp / mix_MW
    p_part = {sp: C_b[sp]*R_SI*T_b/1e5 for sp in SPECIES}
    p_tot = sum(p_part[sp] for sp in SPECIES) # bar
    rho_g = p_tot * mix_MW / R_SI / T_b * 1e5 / 1000    # kg/m^3
    conv_T = u_s * rho_g * cp * 1000 * (T_b - T_prev) / dz
    htc_b = 250 # 1 / (1 / h_w + R_t / 4 / k_er_f)
    ext_T = 2 * htc_b / R_t * (T_b - T_wall)
    res[_N_SP] = conv_T - inter_T + ext_T
    
    # ============================================================
    # Pellet equations
    # ============================================================
    poff = _N_SP + 1
    for p in range(_N_P):
        if p == 0:
            # Mass balance
            for i, sp in enumerate(SPECIES):
                diff = sum(_Ap[p, m] * C_s[sp][m] for m in range(_N_P))
                res[poff+i*_N_P+p] = diff
            
            # Energy balance
            diff_T = sum(_Ap[p, m] * T_s[m] for m in range(_N_P))
            res[-_N_P+p] = diff_T
        # ---------- Center and interior pellet points: PDE ----------
        elif p > 0 and p < _N_P - 1:
            p_part = {sp: C_s[sp][p]*R_SI*T_s[p]/1e5 for sp in SPECIES}
            rxn = reaction_rates(T_s[p], p_part, a_node)
            S = species_source_terms(rxn)
            for sp in S:
                S[sp] *= rho_p * 1000.0 / 3600.0    # mol/(m^3·s)
            Q = heat_source_term(rxn)
            Q *= rho_p * 1000.0 / 3600.0   # W/m^3
        
            # Mass balance
            for i, sp in enumerate(SPECIES):
                diff = sum(_Mp[p, m] * C_s[sp][m] for m in range(_N_P))
                diff *= D_e[SPECIES[i]] / R_p**2
                res[poff+i*_N_P+p] = diff + S[sp]
            
            # Energy balance
            diff_T = sum(_Mp[p, m] * T_s[m] for m in range(_N_P))
            diff_T *= k_e_s / R_p**2
            res[-_N_P+p] = diff_T + Q
        else:
            # ---------- Pellet surface: flux-matching BC ----------
            for i, sp in enumerate(SPECIES):
                dCdx = sum(_Ap[_N_P - 1, m] * C_s[sp][m] for m in range(_N_P))
                flux_int = (D_e[SPECIES[i]] / R_p) * dCdx
                flux_ext = k_c * (C_s[sp][_N_P - 1] - C_b[sp])
                res[poff+i*_N_P+p] = flux_int + flux_ext
            dTdx = sum(_Ap[_N_P - 1, m] * T_s[m] for m in range(_N_P))
            flux_T_int = (k_e_s / R_p) * dTdx
            flux_T_ext = h_f * (T_s[_N_P - 1] - T_b)
            res[-1] = flux_T_int + flux_T_ext
    
    return res        


def _residual_homo_2d(yraw, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node):
    y = np.exp(yraw)
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
        if k == 0:
            for i, sp in enumerate(SPECIES):
                res[i*_N_K] = sum(_Ar[0, m] * Cj[sp][m] for m in range(_N_K))
            dTdx = sum(_Ar[0, m] * Tj[m] for m in range(_N_K))
            res[-_N_K] = dTdx
        elif k > 0 and k < _N_K - 1:
            a_loc = a_node[k]
            p_part = {sp: Cj[sp][k]*R_SI*Tj[k]/1e5 for sp in SPECIES}
            p_tot = sum(p_part[sp] for sp in SPECIES) # bar
            rxn = reaction_rates(Tj[k], p_part, a_loc)
            S = species_source_terms(rxn)
            for sp in S:
                S[sp] *= rho_b * 1000.0 / 3600.0    # mol/(m^3·s)
            Q = heat_source_term(rxn)
            Q *= rho_b * 1000.0 / 3600.0   # W/m^3
            for i, sp in enumerate(SPECIES):
                # Radial diffusion
                rad = (D_er * 4.0/R_t**2) * sum(_Mr[k, m]*Cj[sp][m]
                                                for m in range(_N_K))
                # Axial convection (backward difference for stability)
                conv = u_s * (Cj[sp][k] - C_prev[sp][k]) / dz
                res[i*_N_K+k] = rad - conv + S[sp]
            # Energy
            sum_C_cp = sum(Cj[SPECIES[i]][k] / C_total_in * CP_MOL[SPECIES[i]] 
                           for i in range(_N_SP))
            mix_MW = sum(Cj[SPECIES[i]][k] / C_total_in * MW[SPECIES[i]] 
                         for i in range(_N_SP))
            cp = sum_C_cp / mix_MW
            rho_g = p_tot * mix_MW / R_SI / Tj[k] * 1e5 / 1000
            rad_T = (k_er_f * 4.0/R_t**2) * sum(_Mr[k, m]*Tj[m]
                                            for m in range(_N_K))
            conv_T = u_s * rho_g * cp * 1000 * (Tj[k] - T_prev[k]) / dz
            res[-_N_K+k] = rad_T - conv_T + Q
        else:
            # ---------- Wall point (k = N_K - 1) ----------
            # Mass: zero flux at wall -> dC/dx = 0
            for i, sp in enumerate(SPECIES):
                res[i*_N_K+k] = sum(_Ar[_N_K - 1, m] * Cj[sp][m]
                                    for m in range(_N_K))
            # Energy: (2 k_er / R_t) dT/dx|_wall = h_w (T_wall - T_b[wall])
            dTdx = sum(_Ar[_N_K - 1, m] * Tj[m] for m in range(_N_K))
            res[-1] = ((2.0 * k_er_f / R_t) * dTdx + h_w * (Tj[_N_K - 1] - T_wall))
        
    return res


def _residual_homo_1d(yraw, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node):
    y = np.exp(yraw)
    S = {sp: 0.0 for sp in SPECIES}
    Q = 0.0
    C_b = {sp: 0.0 for sp in SPECIES}
    
    for i, sp in enumerate(SPECIES):
        C_b[sp] = y[i]
    T_b = y[_N_SP]
    
    n_unk = (_N_SP + 1)
    res = np.zeros(n_unk)
    
    # ============================================================
    # Fluid-phase equations
    # ============================================================
    p_part = {sp: C_b[sp]*R_SI*T_b/1e5 for sp in SPECIES}
    p_tot = sum(p_part[sp] for sp in SPECIES) # bar
    rxn = reaction_rates(T_b, p_part, a_node)
    S = species_source_terms(rxn)
    for sp in S:
        S[sp] *= rho_b * 1000.0 / 3600.0    # mol/(m^3·s)
    Q = heat_source_term(rxn)
    Q *= rho_b * 1000.0 / 3600.0   # W/m^3
    # Mass balance for each species
    for i, sp in enumerate(SPECIES):
        # inter = -k_c * a_v * (C_b[sp] - C_s[sp][_N_P - 1])
        conv = u_s * (C_b[sp] - C_prev[sp]) / dz
        source = S[sp]
        res[i] = conv - source # + inter 
    # Energy balance
    # inter_T = h_f * a_v * (T_b - T_s[_N_P - 1])
    sum_C_cp = sum(C_b[SPECIES[i]] / C_total_in * CP_MOL[SPECIES[i]] 
                   for i in range(_N_SP))
    mix_MW = sum(C_b[SPECIES[i]] / C_total_in * MW[SPECIES[i]] 
                 for i in range(_N_SP))
    cp = sum_C_cp / mix_MW  # kJ/(Kg.K)
    rho_g = p_tot * mix_MW / R_SI / T_b * 1e5 / 1000    # kg/m^3
    conv_T = u_s * rho_g * cp * 1000 * (T_b - T_prev) / dz
    source_T = Q
    htc_b = 250 # 1 / (1 / h_w + R_t / 4 / k_er_f)
    ext_T = 2 * htc_b / R_t * (T_b - T_wall)
    res[_N_SP] = conv_T - source_T + ext_T # + inter_T 
    
    return res

    
def _ode_homo_1d(t, y, T_wall, a_node):
    S = {sp: 0.0 for sp in SPECIES}
    Q = 0.0
    C_b = {sp: 0.0 for sp in SPECIES}
    
    for i, sp in enumerate(SPECIES):
        C_b[sp] = y[i]
    T_b = y[_N_SP]
    
    n_unk = (_N_SP + 1)
    dy = np.zeros(n_unk)
    
    # ============================================================
    # Fluid-phase equations
    # ============================================================
    p_part = {sp: C_b[sp]*R_SI*T_b/1e5 for sp in SPECIES}
    p_tot = sum(p_part[sp] for sp in SPECIES) # bar
    rxn = reaction_rates(T_b, p_part, a_node)
    S = species_source_terms(rxn)
    for sp in S:
        S[sp] *= rho_b * 1000.0 / 3600.0    # mol/(m^3·s)
    Q = heat_source_term(rxn)
    Q *= rho_b * 1000.0 / 3600.0   # W/m^3
    # Mass balance for each species
    for i, sp in enumerate(SPECIES):
        source = S[sp]
        dy[i] = source / u_s # 
    # Energy balance
    sum_C_cp = sum(C_b[SPECIES[i]] / C_total_in * CP_MOL[SPECIES[i]] 
                   for i in range(_N_SP))
    mix_MW = sum(C_b[SPECIES[i]] / C_total_in * MW[SPECIES[i]] 
                 for i in range(_N_SP))
    cp = sum_C_cp / mix_MW  # kJ/(Kg.K)
    rho_g = p_tot * mix_MW / R_SI / T_b * 1e5 / 1000    # kg/m^3
    source_T = Q
    Utc = 250 # 1 / (1 / h_w + R_t / 4 / k_er_f)
    ext_T = 2 * Utc / R_t * (T_b - T_wall)
    dy[_N_SP] = (source_T - ext_T) / (u_s * cp * rho_g * 1000) # + inter_T 
    
    return dy


def solve_reactor_2d(
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
    C_pellet = {sp: np.zeros((Nz + 1, _N_K)) for sp in SPECIES}
    T_pellet = np.zeros((Nz + 1, _N_K))
    
    # ---- Inlet conditions ----
    for sp in SPECIES:
        C_bulk[sp][0, :] = C_in[sp]
        C_pellet[sp][0, :] = C_in[sp]
    T_bulk[0, :] = T_in
    T_pellet[0, :] = T_in
    
    # ---- Initial guess ----
    if heterogeneous:
        y_guess = _initial_guess_hetero_2d(C_in, T_in)
    else:
        y_guess = _initial_guess_homo_2d(C_in, T_in)
        
    for j in range(1, Nz + 1):
        C_prev = {sp: C_bulk[sp][j - 1, :] for sp in SPECIES}
        T_prev = T_bulk[j - 1, :].copy()
        a_node = activity_profile[j, :]
        
        def residual(y):
            if heterogeneous:
                return _residual_hetero_2d(
                    y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node
                )
            else:
                return _residual_homo_2d(
                    y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node
                )
        
        sol = root(
            residual, 
            y_guess, 
            method='hybr', 
            options={'xtol': tol, 'maxfev': maxfev},
        )
        
        if not sol.success:
            if verbose:
                print(f"  [!] Node j={j} (z={z[j]:.3f} m) not converged: "
                      f"{sol.message}")
            # Fallback: retry with a smaller trust region
            sol = root(residual, y_guess, method='lm',
                       options={'xtol': tol, 'maxiter': maxfev})
        
        y_guess = sol.x.copy()
        
        if heterogeneous:
            _unpack_hetero_2d(sol.x, j, C_bulk, T_bulk, C_pellet, T_pellet)
        else:
            _unpack_homo_2d(sol.x, j, C_bulk, T_bulk)
            
        if verbose and (j % max(1, Nz // 10) == 0 or j == Nz):
            T_avg = T_bulk[j, :].mean()
            print(f"  z = {z[j]:6.2f} m | "
                  f"<T_bulk> = {T_avg:7.2f} K | "
                  f"a_min = {a_node.min():.4f}")
    
    # eta = compute_effectiveness_factors(
    #     C_bulk, T_bulk, C_pellet, T_pellet, activity_profile
    # )

    return C_bulk, T_bulk, C_pellet, T_pellet


def solve_reactor_1d(
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
    activity_profile : ndarray, shape (Nz+1)
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
    C_bulk   : dict of ndarray (Nz+1)   fluid bulk concentrations
    T_bulk   : ndarray (Nz+1)           fluid bulk temperatures
    C_pellet : dict of ndarray (Nz+1, N_P)  pellet concentrations
    T_pellet : ndarray (Nz+1, N_P)      pellet temperatures
    """
    Nz = len(z) - 1
    
    C_bulk   = {sp: np.zeros(Nz + 1) for sp in SPECIES}
    T_bulk   = np.zeros(Nz + 1)
    C_pellet = {sp: np.zeros((Nz + 1, _N_P)) for sp in SPECIES}
    T_pellet = np.zeros((Nz + 1, _N_P))
    
    # ---- Inlet conditions ----
    for sp in SPECIES:
        C_bulk[sp][0] = C_in[sp]
        C_pellet[sp][0, :] = C_in[sp]
    T_bulk[0] = T_in
    T_pellet[0, :] = T_in
    
    # ---- Initial guess ----
    if heterogeneous:
        y_guess = _initial_guess_hetero_1d(C_in, T_in)
    else:
        y_guess = _initial_guess_homo_1d(C_in, T_in)
        
    for j in range(1, Nz + 1):
        C_prev = {sp: C_bulk[sp][j - 1] for sp in SPECIES}
        T_prev = T_bulk[j - 1].copy()
        a_node = activity_profile[j]
        
        def residual(y):
            if heterogeneous:
                return _residual_hetero_1d(
                    y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node
                )
            else:
                return _residual_homo_1d(
                    y, C_prev, T_prev, C_in, T_in, T_wall, dz, a_node
                )
        
        sol = root(
            residual, 
            y_guess, 
            method='hybr', 
            options={'xtol': tol, 'maxfev': maxfev},
        )
        
        if not sol.success:
            if verbose:
                print(f"  [!] Node j={j} (z={z[j]:.3f} m) not converged: "
                      f"{sol.message}")
            # Fallback: retry with a smaller trust region
            sol = root(residual, y_guess, method='lm',
                       options={'xtol': tol, 'maxiter': maxfev})
        
        y_guess = sol.x.copy()
        
        if heterogeneous:
            _unpack_hetero_1d(sol.x, j, C_bulk, T_bulk, C_pellet, T_pellet)
        else:
            _unpack_homo_1d(sol.x, j, C_bulk, T_bulk)
            
        if verbose and (j % max(1, Nz // 10) == 0 or j == Nz):
            T_avg = T_bulk[j]
            print(f"  z = {z[j]:6.2f} m | "
                  f"<T_bulk> = {T_avg:7.2f} K | "
                  f"a_min = {a_node.min():.4f}")
    
    # eta = compute_effectiveness_factors(
    #     C_bulk, T_bulk, C_pellet, T_pellet, activity_profile
    # )

    return C_bulk, T_bulk, C_pellet, T_pellet


def solve_reactor_1d_ode(
    z, dz, activity_profile, C_in, T_in, T_wall, verbose=True
):
    """
    Solve ther reactor model through ODE.

    Parameters
    ----------
    z : ndarray, shape (Nz+1,)
        Axial grid points [m].
    dz : float
        Axial step size [m].
    activity_profile : ndarray, shape (Nz+1)
        Catalyst activity at each axial and radial location [-].
    C_in : dict
        Inlet concentrations [mol/m^3] keyed by SPECIES names.
    T_in : float
        Inlet temperature [K].
    T_wall : float
        Wall temperature [K].
    verbose : bool
        Print progress.

    Returns
    -------
    C_bulk   : dict of ndarray (Nz+1)   fluid bulk concentrations
    T_bulk   : ndarray (Nz+1)           fluid bulk temperatures
    C_pellet : dict of ndarray (Nz+1, N_P)  pellet concentrations
    T_pellet : ndarray (Nz+1, N_P)      pellet temperatures
    """
    Nz = len(z) - 1
    
    C_bulk   = {sp: np.zeros(Nz + 1) for sp in SPECIES}
    T_bulk   = np.zeros(Nz + 1)
    C_pellet = {sp: np.zeros((Nz + 1, _N_P)) for sp in SPECIES}
    T_pellet = np.zeros((Nz + 1, _N_P))
    
    # ---- Inlet conditions ----
    for sp in SPECIES:
        C_bulk[sp][0] = C_in[sp]
        C_pellet[sp][0, :] = C_in[sp]
    T_bulk[0] = T_in
    T_pellet[0, :] = T_in
    
    for j in range(1, Nz + 1):
        C_prev = {sp: C_bulk[sp][j - 1] for sp in SPECIES}
        T_prev = T_bulk[j - 1].copy()
        a_node = activity_profile[j]
        
        sol = solve_ivp(
            _ode_homo_1d, 
            [0, dz], 
            [C_prev[sp] for sp in SPECIES] + [T_prev], 
            args=(T_wall, a_node), 
            method="BDF",
            # t_eval=z,
        )
    
        for i, sp in enumerate(SPECIES):
            C_bulk[sp][j] = sol.y[i, -1]
        T_bulk[j] = sol.y[_N_SP, -1]
        
        if verbose and (j % max(1, Nz // 10) == 0 or j == Nz):
            T_avg = T_bulk[j]
            print(f"  z = {z[j]:6.2f} m | "
                  f"<T_bulk> = {T_avg:7.2f} K | "
                  f"a_min = {a_node.min():.4f}")

    return C_bulk, T_bulk, C_pellet, T_pellet