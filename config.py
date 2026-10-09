# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:55:19 2026

@author: nazmo
"""

# ---------- Physical constants ----------
R_gas = 8.314e-3          # kJ/(mol·K)
R_SI  = 8.314             # J/(mol·K)

# ---------- Reactor geometry ----------
L      = 12.0             # Reactor length [m]
R_t    = 0.05             # Tube inner radius [m]
R_p    = 0.003            # Pellet radius [m]

# ---------- Bed properties ----------
eps_bed   = 0.40          # Bed voidage [-]
rho_b     = 1200.0        # Bed density [kg/m^3]
rho_p     = 2000.0        # Pellet density [kg/m^3]
a_v       = 3.0 * (1 - eps_bed) / R_p   # Specific surface area [m^2/m^3]

# ---------- Transport properties ----------
D_er      = 5.0e-4        # Effective radial diffusivity [m^2/s]
k_er_f    = 0.5           # Effective radial thermal conductivity (fluid) [W/(m·K)]
k_e_s     = 1.5           # Effective thermal conductivity (pellet) [W/(m·K)]
D_e_CH4   = 1.0e-7        # Effective diffusivity CH4 in pellet [m^2/s]
D_e_H2O   = 1.5e-7
D_e_CO    = 1.2e-7
D_e_CO2   = 1.0e-7
D_e_H2    = 4.0e-7

# ---------- Heat/mass transfer coefficients ----------
h_f       = 200.0         # Fluid-solid heat transfer [W/(m^2·K)]
h_w       = 500.0         # Wall heat transfer [W/(m^2·K)]
k_c       = 0.05          # External mass transfer [m/s]

# ---------- Operating conditions ----------
u_s       = 1.5           # Superficial velocity [m/s]
T_in      = 800.0 + 273.15
T_wall    = 1150.0 + 273.15
P_total   = 25.0e5        # Pa

# Inlet mole fractions (typical industrial SMR)
y_in = {
    'CH4': 0.20, 'H2O': 0.60, 'H2': 0.10,
    'CO':  0.02, 'CO2': 0.05, 'N2': 0.03
}
C_total_in = P_total / (R_SI * T_in)      # mol/m^3

# ---------- Discretization ----------
N_r  = 6                  # Interior radial collocation points
N_p  = 10                 # Interior pellet collocation points
N_z  = 100                # Axial finite-difference nodes

# ---------- Time stepping ----------
t_final     = 720 * 3600.0    # 720 hours
dt_deact    = 1 * 3600.0      # 1 hour per deactivation step

# ---------- Xu & Froment kinetic parameters ----------
kinetics = {
    'k_SMR': 1.842e-4, 'E_SMR': 240.1,
    'k_WGS': 7.558,    'E_WGS': 67.13,
    'k_GRR': 2.193e-5, 'E_GRR': 243.9,
    'K_CH4': 0.1791,   'dH_CH4': -38.28,
    'K_CO':  40.91,    'dH_CO':  -70.65,
    'K_H2':  0.0296,   'dH_H2':  -82.90,
    'K_H2O': 0.4152,   'dH_H2O':  88.68,
    'T_ref': 648.0,
}

# ---------- Deactivation parameters ----------
deact = {
    'k_d0':  1.0e-3,       # Pre-exponential [1/h]
    'E_d':   50.0,         # kJ/mol
    'order': 1.0,          # d in da/dt = -k_d a^d C_coke^n
    'n_coke': 1.0,
    'a_ss':  0.05,         # Residual activity
    'k_MD':  1.0e-4,       # Methane decomposition rate constant
    'E_MD':  100.0,
    'k_Boud': 5.0e-5,
    'E_Boud': 80.0,
}