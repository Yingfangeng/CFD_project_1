# ================================================
# Imperial College London
# Department of Mechaincal Engineering
# Computational Fluid Dynamics Coursework
# A Comparison Study of Difference Schemes in Solving 1D Convection-Diffusion Equations of a Generic Property
# The Solver Code
# Author: Yingfan Geng
# CID: 02195246
# January 2026
# ================================================


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def tdma(phi_0, phi_L, rho, u, gamma, L, a, b, c, number_of_grid):

    delta_x = L/(number_of_grid-1)      # first and last grid is half-cell, so full cell number is N-1
    
    # initialise the P, Q and phi lists
    Q_list = np.zeros(number_of_grid)
    P_list = np.zeros(number_of_grid)
    phi_list = np.zeros(number_of_grid)
    
    
    # apply the boundary conditions
    Q_list[0] = phi_0
    P_list[0] = 0
    Q_list[number_of_grid-1] = phi_L
    P_list[number_of_grid-1] = 0
    phi_list[0] = phi_0
    phi_list[number_of_grid-1] = phi_L

    # forward sweep to calculate P and Q at each node
    for i in range(1, number_of_grid-1):
        P_list[i] = b / (a - c*P_list[i-1])
        Q_list[i] = c*Q_list[i-1] / (a - c*P_list[i-1])

    # backward sweep to calculate phi at each node
    for i in reversed(range(1, number_of_grid-1)):
        phi_list[i] = P_list[i]*phi_list[i+1] + Q_list[i]

    return phi_list



def analytical_phi(x, L, rho, u, gamma, phi_0, phi_L):
    
    phi_analytic = phi_0 + (((np.exp(rho*u*x/gamma))-1)/((np.exp(rho*u*L/gamma))-1)) *(phi_L - phi_0)

    return phi_analytic



def solver(number_of_grid, L, rho, u, gamma, phi_0, phi_L, scheme):

    x = np.linspace(0, L, number_of_grid)
    delta_x = x[1] - x[0]
    Pe_local = rho * u * delta_x / gamma
    Pe_global = rho * u * L / gamma
    print(f'The local Peclet number is {Pe_local}, global Pecklet number is {Pe_global}.')

    if scheme == "CDS":
        a_E = gamma/delta_x - rho*u/2
        a_W = gamma/delta_x + rho*u/2
        a_P = a_E + a_W

    elif scheme == "UDS":
        a_E = gamma/delta_x + max(-(rho*u), 0)
        a_W = gamma/delta_x + max(rho*u, 0)
        a_P = a_E + a_W

    elif scheme == "PLDS":
        
        a_E = gamma/delta_x * max(0, (1 - 0.1 * abs(Pe_local)) ** 5) + max(-(rho*u), 0)
        a_W = gamma/delta_x * max(0, (1 - 0.1 * abs(Pe_local)) ** 5) + max(rho*u, 0)
        a_P = a_E + a_W
    else:
        raise NotImplementedError

    phi = tdma(phi_0, phi_L, rho, u, gamma, L, a_P, a_E, a_W, number_of_grid)

    return phi, Pe_local



def error_calculation(phi_actual, phi_analytic):
    
    error = 100 * np.mean(np.abs((phi_actual - phi_analytic) / phi_analytic))
    
    return error



# =================================================================
# ========================= Visualisation =========================
# =================================================================

# set the font
text_font = 'Arial'
math_font = 'stix'
plt.rcParams['mathtext.fontset'] = math_font
rcParams['font.family'] = text_font


def save_fig_custom(fig, file_path='', file_name='fig', format_list=['.eps', '.png'], overwrite=False, dpi = 500):
    
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



def plot_the_result(L, rho, gamma, phi_0, phi_L, number_of_grid, u, scheme_list):
    
    x = np.linspace(0, L, number_of_grid)
    phi_analytic = analytical_phi(x, L, rho, u, gamma, phi_0, phi_L)
    
    # plot the analytical solution smoothly
    x_for_analytic_plot = np.linspace(0, L, 999)
    phi_analytic_plot = analytical_phi(x_for_analytic_plot, L, rho, u, gamma, phi_0, phi_L)
    
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=plt.cm.Set1.colors)
    ax.plot(x_for_analytic_plot, phi_analytic_plot, label = 'analytic')
    
    for scheme in scheme_list: 
        phi_actual, Pe_local = solver(number_of_grid, L, rho, u, gamma, phi_0, phi_L, scheme)
        ax.plot(x, phi_actual, marker = 'o', label = f'{scheme}')
    ax.set_xlim(0,1)
    ax.set_ylim(15,105)
    ax.grid(ls = ':')
    ax.legend(loc = 'lower left', fontsize = 14)
    ax.set_xlabel('x (m)', fontsize = 14)
    ax.set_ylabel('$\phi$', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    save_fig_custom(fig, file_path='figures', file_name=f'phi_plot_grid_{number_of_grid}_u_{u}_pe_{Pe_local:.2f}', overwrite=True, dpi = 500)


def error_plot(L, rho, gamma, phi_0, phi_L, u, number_of_grid_list, scheme_list):
    

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=plt.cm.Set1.colors)
    
    for scheme in scheme_list: 
        error_list = []
        delta_x_list = []
        for number_of_grid in number_of_grid_list:
            x = np.linspace(0, L, number_of_grid)
            delta_x = x[1] - x[0]
            delta_x_list.append(delta_x)
            phi_analytic = analytical_phi(x, L, rho, u, gamma, phi_0, phi_L)
            phi_actual, Pe_local = solver(number_of_grid, L, rho, u, gamma, phi_0, phi_L, scheme)
            error = error_calculation(phi_actual, phi_analytic)
            error_list.append(error)
        ax.plot(delta_x_list, error_list, label = f'{scheme}')
    ax.grid(ls = ':')
    ax.legend(loc = 'upper left', fontsize = 14)
    ax.set_xlabel('$\delta x$ (m)', fontsize = 14)
    ax.set_ylabel('Percentage Mean Error', fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    save_fig_custom(fig, file_path='figures', file_name=f'error_plot_u_{u}_pe_{Pe_local:.2f}', overwrite=True, dpi = 500)

