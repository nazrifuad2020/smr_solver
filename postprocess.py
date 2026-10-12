# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:58:27 2026

@author: nazmo
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_profiles(history, z, xr):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # Activity vs axial position (center and wall)
    for idx, (label, k) in enumerate([('center', 0), ('wall', -1)]):
        ax[0, 0].plot(z, history['a'][0][:, k], '--',
                      label=f't=0 {label}')
        ax[0, 0].plot(z, history['a'][-1][:, k], '-',
                      label=f't=end {label}')
    ax[0, 0].set_xlabel('z [m]'); ax[0, 0].set_ylabel('Activity a')
    ax[0, 0].legend(); ax[0, 0].grid(True)

    # Temperature profile at end
    T_end = history['T'][-1]
    for k in [0, len(xr)//2, -1]:
        ax[0, 1].plot(z, T_end[:, k], label=f'r/R_t = {xr[k]**0.5:.2f}')
    ax[0, 1].set_xlabel('z [m]'); ax[0, 1].set_ylabel('T [K]')
    ax[0, 1].legend(); ax[0, 1].grid(True)

    # Methane conversion vs time
    t_h = np.array(history['t'])
    ax[1, 0].plot(t_h, history['X_CH4'], 'b-o')
    ax[1, 0].set_xlabel('t [h]'); ax[1, 0].set_ylabel('X_CH4')
    ax[1, 0].grid(True)

    # Coke concentration
    Cc_end = history['C_coke'][-1]
    im = ax[1, 1].imshow(Cc_end.T, aspect='auto', origin='lower',
                         extent=[0, z[-1], 0, 1])
    ax[1, 1].set_xlabel('z [m]'); ax[1, 1].set_ylabel('r/R_t')
    ax[1, 1].set_title('Coke concentration')
    plt.colorbar(im, ax=ax[1, 1])

    plt.tight_layout()
    plt.savefig('smr_deactivation.png', dpi=150)