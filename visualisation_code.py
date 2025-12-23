# ================================================
# Imperial College London
# Department of Mechaincal Engineering
# Computational Fluid Dynamics Coursework
# A Comparison Study of Difference Schemes in Solving 1D Convection-Diffusion Equations of a Generic Property
# The Visualisation Code
# Author: Yingfan Geng
# CID: 02195246
# January 2026
# ================================================

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
from solver_code_new import *


# set the font
text_font = 'Arial'
math_font = 'stix'
plt.rcParams['mathtext.fontset'] = math_font
rcParams['font.family'] = text_font


def save_fig_custom(fig, file_path='', file_name='fig', format_list=['.eps', '.png'], overwrite=False, dpi = 500):
    
    '''
    Save the plotted graph as .esp and .png in high resolution.
    '''

    if (file_path != '') and (file_path[-1] != '/'): 
        file_path = file_path+'/'
    
    if (file_path != '') and (os.path.isdir(file_path) == False):
        os.makedirs(file_path)
        print('Save directory %s is created'%(file_path))
    
    for save_format in format_list:
        if save_format[0] != '.': save_format = '.' + save_format

        file_name_now = file_path+file_name+save_format
        if not overwrite:
            i = 1
            while os.path.exists(file_name_now):
                file_name_now = file_path+file_name + '%i'%(i)+save_format
                i = i+1
        
        if save_format == '.png':
            fig.savefig(file_name_now, facecolor='white', bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(file_name_now, facecolor='white', bbox_inches="tight")

        print(file_name_now, 'is saved.')
    
    return



def plot_the_result(L, rho, gamma, phi_0, phi_L, number_of_grid, u, scheme):
    
    x = np.linspace(0, L, number_of_grid)
    phi_analytic = analytical_phi(x, L, rho, u, gamma, phi_0, phi_L)
    phi_actual = solver(number_of_grid, L, rho, u, gamma, phi_0, phi_L, scheme)
    error = error_calculation(phi_actual, phi_analytic)
    fig, ax = plt.subplots()
    ax.plot(x, phi_analytic, label = 'analytic')
    ax.plot(x, phi_actual, marker = 'o', label = f'{scheme}')
    ax.set_xlim(0,1)
    ax.grid(ls = ':')
    ax.legend(fontsize = 14)
    ax.set_xlabel('x(m)', fontsize = 14)
    ax.set_ylabel('$\phi$', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    save_fig_custom(fig, file_path='figures', file_name='result', overwrite=True, dpi = 500)