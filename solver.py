# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:57:46 2026

@author: nazmo
"""

import numpy as np
from scipy.integrate import solve_ivp
from config import (L, N_z, N_r, t_final, dt_deact, C_total_in, y_in,
                    T_in, T_wall, u_s)
from grid import axial_grid, reactor_radial_grid
from reactor import solve_reactor_1d, solve_reactor_2d, solve_reactor_1d_ode
from deactivation import coke_formation_rate, activity_ode
from pellet import compute_effectiveness_factors


def initialize_activity():
    """a(z,r) = 1 everywhere at t=0."""
    _, _, _, _ = reactor_radial_grid(N_r)
    return np.ones((N_z + 1, N_r + 2))

def run_simulation_2d(dynamic=True, heterogeneous=True):
    z, dz = axial_grid(L, N_z)
    xr, _, _, _ = reactor_radial_grid(N_r)
    n_r = len(xr)

    # Inlet concentrations [mol/m^3]
    C_in = {sp: y_in[sp] * C_total_in for sp in y_in}

    # Initial activity
    a = initialize_activity()

    # Store history
    history = {'t': [], 'a': [], 'T': [], 'X_CH4': [], 'C_coke': []}

    t = 0.0
    C_coke = np.zeros_like(a)          # Coke concentration field

    tf = t_final
    if not dynamic:
        tf = dt_deact

    while t < tf:
        # --- Solve steady-state reactor with current activity ---
        C, T, Cp, Tp = solve_reactor_2d(
            z, dz, a, C_in, T_in, T_wall, heterogeneous=heterogeneous
        )

        if dynamic:
            # --- Compute local coke formation rate ---
            r_coke = np.zeros_like(a)
            for j in range(N_z + 1):
                for k in range(n_r):
                    p = {sp: C[sp][j, k] * 8.314 * T[j, k] / 1e5
                         for sp in C}   # bar
                    r_coke[j, k] = coke_formation_rate(T[j, k], p, a[j, k])
                
            # --- Integrate deactivation ODE over dt_deact ---
            def rhs(tau, state):
                a_flat = state[:a.size].reshape(a.shape)
                Cc_flat = state[a.size:].reshape(a.shape)
                da = np.zeros_like(a_flat)
                dCc = np.zeros_like(Cc_flat)
                for j in range(N_z + 1):
                    for k in range(n_r):
                        da[j, k] = activity_ode(a_flat[j, k], Cc_flat[j, k],
                                                T[j, k])
                        dCc[j, k] = r_coke[j, k]
                return np.concatenate([da.ravel(), dCc.ravel()])

            state0 = np.concatenate([a.ravel(), C_coke.ravel()])
            sol = solve_ivp(rhs, [0, dt_deact], state0, method='BDF',
                            rtol=1e-6, atol=1e-9)
            a = sol.y[:a.size, -1].reshape(a.shape)
            C_coke = sol.y[a.size:, -1].reshape(a.shape)
            
            history['C_coke'].append(C_coke.copy())

        # --- Compute methane conversion ---
        F_CH4_in = u_s * np.pi * 0.05**2 * C_in['CH4']
        F_CH4_out = u_s * np.pi * 0.05**2 * np.mean(C['CH4'][-1, :])
        X_CH4 = 1.0 - F_CH4_out / F_CH4_in

        # --- Store history ---
        history['t'].append(t)
        history['a'].append(a.copy())
        history['T'].append(T.copy())
        history['X_CH4'].append(X_CH4)
        history['C_coke'].append(C_coke.copy())

        print(f"t = {t:6.1f} h | X_CH4 = {X_CH4:.4f} | "
              f"a_min = {a.min():.4f} | C_coke_ave = {C_coke.mean():.4f}")

        t += dt_deact

    return history, z, xr

def run_simulation_1d(dynamic=True, heterogeneous=True, ode=False):
    z, dz = axial_grid(L, N_z)
    xr = None

    # Inlet concentrations [mol/m^3]
    C_in = {sp: y_in[sp] * C_total_in for sp in y_in}

    # Initial activity
    a = np.ones(N_z + 1)

    # Store history
    history = {'t': [], 'a': [], 'T': [], 'X_CH4': [], 'C_coke': [], 'eta': []}

    t = 0.0
    C_coke = np.zeros_like(a)          # Coke concentration field

    tf = t_final
    if not dynamic:
        tf = dt_deact

    while t < tf:
        # --- Step 1: Solve steady-state reactor with current activity ---
        if not ode:
            C, T, Cp, Tp = solve_reactor_1d(
                z, dz, a, C_in, T_in, T_wall, heterogeneous=heterogeneous
            )
        else:
            C, T, Cp, Tp = solve_reactor_1d_ode(
                z, dz, a, C_in, T_in, T_wall
            )

        # --- Step 2: Compute local coke formation rate ---
        r_coke = np.zeros_like(a)
        for j in range(N_z + 1):
            p = {sp: C[sp][j] * 8.314 * T[j] / 1e5 for sp in C}   # bar
            r_coke[j] = coke_formation_rate(T[j], p, a[j])
                
        # --- Step 3: Integrate deactivation ODE over dt_deact ---
        def rhs(tau, state):
            a_flat = state[:a.size].reshape(a.shape)
            Cc_flat = state[a.size:].reshape(a.shape)
            da = np.zeros_like(a_flat)
            dCc = np.zeros_like(Cc_flat)
            for j in range(N_z + 1):
                da[j] = activity_ode(a_flat[j], Cc_flat[j], T[j])
                dCc[j] = r_coke[j]
            return np.concatenate([da.ravel(), dCc.ravel()])

        state0 = np.concatenate([a.ravel(), C_coke.ravel()])
        sol = solve_ivp(rhs, [0, dt_deact], state0, method='BDF',
                        rtol=1e-6, atol=1e-9)
        a = sol.y[:a.size, -1].reshape(a.shape)
        C_coke = sol.y[a.size:, -1].reshape(a.shape)
        
        if heterogeneous:
            eta = compute_effectiveness_factors(C, T, Cp, Tp, a)

        # --- Step 4: Compute methane conversion ---
        F_CH4_in = u_s * np.pi * 0.05**2 * C_in['CH4']
        F_CH4_out = u_s * np.pi * 0.05**2 * C['CH4'][-1]
        X_CH4 = 1.0 - F_CH4_out / F_CH4_in

        # --- Step 5: Store history ---
        history['t'].append(t)
        history['a'].append(a.copy())
        history['T'].append(T.copy())
        history['X_CH4'].append(X_CH4)
        history['C_coke'].append(C_coke.copy())
        if heterogeneous:
            history['eta'].append(eta.copy())

        print(f"t = {t:6.1f} h | X_CH4 = {X_CH4:.4f} | "
              f"a_min = {a.min():.4f} | C_coke_ave = {C_coke.mean():.4f}")

        t += dt_deact

    return history, z, xr

if __name__ == '__main__':
    history, z, xr = run_simulation_1d(dynamic=False, heterogeneous=True)