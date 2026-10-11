# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:56:43 2026

@author: nazmo
"""

import numpy as np
from config import R_SI, deact as D

def coke_formation_rate(T, p, a):
    """Net coke formation rate [kg_coke/kg_cat/h]."""
    k_MD   = D['k_MD']   * np.exp(-D['E_MD']   / R_SI * (1.0/T - 1.0/900.0) * 1e3)
    k_Boud = D['k_Boud'] * np.exp(-D['E_Boud'] / R_SI * (1.0/T - 1.0/900.0) * 1e3)
    r_MD   = k_MD   * (p['CH4'] - p['H2']**2 / np.exp(-9090 / T + 11.66))
    r_Boud = k_Boud * (p['CO']**2 - p['CO2']  / (p['CO2'] / p['CO']**2))
    
    return max(a * (r_MD + r_Boud), 0.0)

def activity_ode(a, C_coke, T):
    """da/dt = -k_d (a - a_ss)^d C_coke^n."""
    # k_d = D['k_d0'] * np.exp(-D['E_d'] / R_SI * (1.0/T - 1.0/900.0) * 1e3)
    k_d = D['k_d0'] * np.exp(-D['E_d'] / R_SI / T * 1e3)
    return -k_d * (a - D['a_ss'])**D['order'] * C_coke**D['n_coke']