# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:55:59 2026

@author: nazmo
"""

import numpy as np
from config import R_SI, kinetics as K

def k_arrhenius(k_ref, E, T, T_ref):
    return k_ref * np.exp(-E / R_SI * (1.0/T - 1.0/T_ref) * 1e3)

def K_vant_hoff(K_ref, dH, T, T_ref):
    return K_ref * np.exp(-dH / R_SI * (1.0/T - 1.0/T_ref) * 1e3)

def equilibrium_constants(T):
    K_SMR = 1.198e13 * np.exp(-26830.0 / T)
    K_WGS = 1.767e-2 * np.exp( 4400.0 / T)
    K_GRR = 2.117e11 * np.exp(-22430.0 / T)
    return K_SMR, K_WGS, K_GRR

def intrinsic_rates(T, p):
    """p: dict of partial pressures [bar]. Returns r_SMR, r_WGS, r_GRR [kmol/kg_cat/h]."""
    k1 = k_arrhenius(K['k_SMR'], K['E_SMR'], T, K['T_ref'])
    k2 = k_arrhenius(K['k_WGS'], K['E_WGS'], T, K['T_ref'])
    k3 = k_arrhenius(K['k_GRR'], K['E_GRR'], T, K['T_ref'])
    KCH4 = K_vant_hoff(K['K_CH4'], K['dH_CH4'], T, K['T_ref'])
    KCO  = K_vant_hoff(K['K_CO'],  K['dH_CO'],  T, K['T_ref'])
    KH2  = K_vant_hoff(K['K_H2'],  K['dH_H2'],  T, K['T_ref'])
    KH2O = K_vant_hoff(K['K_H2O'], K['dH_H2O'], T, K['T_ref'])

    pH2O, pH2, pCH4 = p['H2O'], p['H2'], p['CH4']
    pCO, pCO2       = p['CO'],  p['CO2']

    DEN = 1.0 + KCO*pCO + KH2*pH2 + KCH4*pCH4 + KH2O*pH2O/pH2

    Keq1, Keq2, Keq3 = equilibrium_constants(T)

    r1 = (k1 / pH2**2.5) * (pCH4*pH2O - pCO*pH2**3 / Keq1) / DEN**2
    r2 = (k2 / pH2)     * (pCO*pH2O  - pCO2*pH2 / Keq2)  / DEN**2
    r3 = (k3 / pH2**3.5)* (pCH4*pH2O**2 - pCO2*pH2**4 / Keq3) / DEN**2
    return r1, r2, r3