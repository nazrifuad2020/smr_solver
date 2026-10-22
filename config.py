# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:55:19 2026

@author: nazmo
"""

"""
config.py
=========
Physical constants, kinetic parameters, operating conditions, and the
stoichiometric matrix for the 2D heterogeneous steam methane reforming
(SMR) reactor model with coke deactivation.
"""

import numpy as np

# ============================================================
# 1. Physical constants
# ============================================================
R_gas = 8.314e-3          # kJ/(mol·K)
R_SI  = 8.314             # J/(mol·K)


# ============================================================
# 2. Reactor geometry
# ============================================================
L      = 12.0             # Reactor length [m]
R_t    = 0.05             # Tube inner radius [m]
R_p    = 0.003            # Pellet radius [m]


# ============================================================
# 3. Bed and pellet properties
# ============================================================
eps_bed   = 0.40          # Bed voidage [-]
rho_b     = 1200.0        # Bed density [kg/m^3]
rho_p     = 2000.0        # Pellet density [kg/m^3]
a_v       = 3.0 * (1 - eps_bed) / R_p   # Specific surface area [m^2/m^3]


# ============================================================
# 4. Transport properties
# ============================================================
D_er      = 5.0e-4        # Effective radial diffusivity [m^2/s]
k_er_f    = 0.5           # Effective radial thermal conductivity (fluid) [W/(m·K)]
k_e_s     = 1.5           # Effective thermal conductivity (pellet) [W/(m·K)]

# Effective diffusivities inside the pellet [m^2/s]
D_e = {
    'CH4': 1.0e-7,
    'H2O': 1.5e-7,
    'CO':  1.2e-7,
    'CO2': 1.0e-7,
    'H2':  4.0e-7,
    'N2':  1.1e-7,
}

# ---------- Heat/mass transfer coefficients ----------
h_f       = 200.0         # Fluid-solid heat transfer [W/(m^2·K)]
h_w       = 500.0         # Wall heat transfer [W/(m^2·K)]
k_c       = 0.05          # External mass transfer [m/s]
U_overall = 279.5         # Overall heat transfer coefficient [W/(m^2·K)]


# ============================================================
# 5. Operating conditions
# ============================================================
u_s       = 1.5           # Superficial velocity [m/s]
T_in      = 550 + 273.15
T_wall    = 1000 + 273.15
P_total   = 25.0e5        # Pa

y_in = {
    'CH4': 0.20, 'H2O': 0.60, 'H2': 0.10,
    'CO':  0.02, 'CO2': 0.05, 'N2': 0.03
}
C_total_in = P_total / (R_SI * T_in)      # mol/m^3


# ============================================================
# 6. Discretization parameters
# ============================================================
N_r  = 6                  # Interior radial collocation points
N_p  = 10                 # Interior pellet collocation points
N_z  = 100                # Axial finite-difference nodes


# ============================================================
# 7. Time stepping
# ============================================================
t_final     = 72    # 720 hours
dt_deact    = 1      # 1 hour per deactivation step


# ============================================================
# 8. Reaction network and stoichiometric matrix
# ============================================================
# Species order used throughout the model
SPECIES = ['CH4', 'H2O', 'CO', 'CO2', 'H2', 'N2']
N_SPECIES = len(SPECIES)

# Reaction order used throughout the model
REACTIONS = ['SMR', 'WGS', 'GRR']
N_REACTIONS = len(REACTIONS)

# Index lookups for fast access
SPECIES_INDEX = {sp: i for i, sp in enumerate(SPECIES)}
REACTION_INDEX = {r: i for i, r in enumerate(REACTIONS)}

# ------------------------------------------------------------
# Stoichiometric matrix NU
#   NU[i, j] = stoichiometric coefficient of species i in reaction j
#   (negative for reactants, positive for products)
#
#   SMR: CH4 + H2O  ⇌  CO + 3 H2
#   WGS: CO  + H2O  ⇌  CO2 + H2
#   GRR: CH4 + 2 H2O ⇌ CO2 + 4 H2
#
#                SMR   WGS   GRR
#   CH4          -1     0    -1
#   H2O          -1    -1    -2
#   CO           +1    -1     0
#   CO2           0    +1    +1
#   H2           +3    +1    +4
#   N2            0     0     0
# ------------------------------------------------------------
NU = np.array([
    [-1,  0, -1],   # CH4
    [-1, -1, -2],   # H2O
    [+1, -1,  0],   # CO
    [ 0, +1, +1],   # CO2
    [+3, +1, +4],   # H2
    [ 0,  0,  0],   # N2 (inert)
], dtype=float)

# Convenience: dictionary view of NU
NU_dict = {
    sp: {r: NU[SPECIES_INDEX[sp], REACTION_INDEX[r]]
         for r in REACTIONS}
    for sp in SPECIES
}

# ------------------------------------------------------------
# Heats of reaction at 298 K [kJ/mol]
# ------------------------------------------------------------
DELTA_H = {
    'SMR': +206.1,
    'WGS':  -41.15,
    'GRR': +165.0,
}

# Vector form (aligned with REACTIONS order)
DELTA_H_vec = np.array([DELTA_H[r] for r in REACTIONS])


# ============================================================
# 9. Xu & Froment kinetic parameters
# ============================================================
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


# ============================================================
# 10. Deactivation and coke formation parameters
# ============================================================
deact = {
    'k_d0':  1.0e-3,       # Pre-exponential [1/h]
    'E_d':   50.0,         # kJ/mol
    'order': 1.0,          # d in da/dt = -k_d (a - a_ss)^d C_coke^n
    'n_coke': 1.0,
    'a_ss':  0.05,         # Residual activity
    'k_MD':  1.0e-4,       # Methane decomposition rate constant
    'E_MD':  100.0,
    'k_Boud': 5.0e-5,
    'E_Boud': 80.0,
    # Equilibrium constants for the driving forces
    'K_MD':   1.0e4,       # [bar]      (placeholder; T-dependent in reality)
    'K_Boud': 1.0e-2,      # [bar^-1]   (placeholder; T-dependent in reality)
    # 'k_MD':  4.22e3,
    # 'E_MD':  88.0,
    # 'k_Boud': 3.15e5,
    # 'E_Boud': 115.4,
}

# Coke-forming reactions (used for coke concentration tracking)
COKE_REACTIONS = ['MD', 'Boud']   # Methane decomposition, Boudouard

# Stoichiometry of coke-forming reactions (carbon as C(s))
#   MD:   CH4 ⇌ C(s) + 2 H2
#   Boud: 2 CO ⇌ C(s) + CO2
NU_COKE = {
    'MD':   {'CH4': -1, 'H2': +2, 'CO': 0, 'CO2': 0, 'H2O': 0, 'N2': 0, 'C': +1},
    'Boud': {'CH4': 0, 'H2': 0, 'CO': -2, 'CO2': +1, 'H2O': 0, 'N2': 0, 'C': +1},
}

# ============================================================
# Molar heat capacities at ~1000 K [J/(mol·K)]
# (Replace with temperature-dependent correlations for accuracy)
# ============================================================
CP_MOL = {
    'CH4': 70.0,
    'H2O': 40.0,
    'CO':  33.0,
    'CO2': 50.0,
    'H2':  30.0,
    'N2':  32.0,
}

MW = {
    'CH4': 16.04,
    'H2O': 18.02,
    'CO':  28.01,
    'CO2': 44.01,
    'H2':  2.02,
    'N2':  28.01,
}

# ============================================================
# Utility functions
# ============================================================
def reaction_rates(T, p, a=1.0):
    """
    Compute the local volumetric rate of each reaction.

    Parameters
    ----------
    T : float
        Local temperature [K].
    p : dict
        Partial pressures of species [bar], keyed by SPECIES names.
    a : float
        Local catalyst activity [-].

    Returns
    -------
    rxn : dict
        Reaction rates [kmol/(kg_cat·h)], keyed by REACTIONS names.
    """
    from kinetics import intrinsic_rates
    r1, r2, r3 = intrinsic_rates(T, p)
    return {'SMR': a * r1, 'WGS': a * r2, 'GRR': a * r3}


def species_source_terms(rxn):
    """
    Compute the net molar source term for each species.

    Parameters
    ----------
    rxn : dict
        Reaction rates keyed by REACTIONS names [kmol/(kg_cat·h)].

    Returns
    -------
    S : dict
        Net source term for each species [kmol/(kg_cat·h)].
        Positive = production, negative = consumption.
    """
    S = {sp: 0.0 for sp in SPECIES}
    for sp in SPECIES:
        i = SPECIES_INDEX[sp]
        for r in REACTIONS:
            j = REACTION_INDEX[r]
            S[sp] += NU[i, j] * rxn[r]
    return S


def heat_source_term(rxn):
    """
    Compute the net heat released/absorbed by the reaction network.

    Parameters
    ----------
    rxn : dict
        Reaction rates keyed by REACTIONS names [kmol/(kg_cat·h)].

    Returns
    -------
    Q : float
        Net heat source [kJ/(kg_cat·h)].
        Negative = endothermic (heat absorbed).
        Positive = exothermic (heat released).
    """
    Q = 0.0
    for r in REACTIONS:
        Q -= DELTA_H[r] * 1000 * rxn[r]     # minus sign: ΔH positive = endothermic
    return Q


def partial_pressures(C, T):
    """
    Convert concentrations [mol/m^3] and temperature [K] to partial
    pressures [bar].

    Parameters
    ----------
    C : dict
        Species concentrations [mol/m^3], keyed by SPECIES names.
    T : float
        Temperature [K].

    Returns
    -------
    p : dict
        Partial pressures [bar], keyed by SPECIES names.
    """
    return {sp: C[sp] * R_SI * T / 1e5 for sp in SPECIES}


def mole_fractions(C):
    """Compute mole fractions from concentrations."""
    C_tot = sum(C.values())
    return {sp: C[sp] / C_tot for sp in SPECIES}


# ============================================================
# 12. Self-consistency check (executed at import)
# ============================================================
def _check_dimensions():
    """Verify that NU, SPECIES, and REACTIONS are dimensionally consistent."""
    assert NU.shape == (N_SPECIES, N_REACTIONS), \
        f"NU shape mismatch: {NU.shape} vs ({N_SPECIES}, {N_REACTIONS})"
    # Check that each reaction conserves mass (sum of NU across species != 0
    # is OK because CH4 + H2O -> CO + 3H2 has different molar balances per
    # element; we only verify no NaNs and no accidental zeros in required slots)
    assert not np.any(np.isnan(NU)), "NU contains NaN"
    for r in REACTIONS:
        j = REACTION_INDEX[r]
        nu_col = NU[:, j]
        # At least one reactant and one product
        assert np.any(nu_col < 0), f"Reaction {r} has no reactants"
        assert np.any(nu_col > 0), f"Reaction {r} has no products"

_check_dimensions()