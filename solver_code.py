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
import time



class convection_diffusion_equation:

    def __init__(self, L, rho, gamma, phi_0, phi_L):
        self.L = L
        self.rho = rho
        self.gamma = gamma
        self.phi_0 = phi_0
        self.phi_L = phi_L



    def solver_tdma(self):
        
        # initialise the P, Q and phi lists
        Q_list = np.zeros(self.number_of_grid)
        P_list = np.zeros(self.number_of_grid)
        phi_list = np.zeros(self.number_of_grid)
        
        # apply the boundary conditions
        Q_list[0] = self.phi_0
        P_list[0] = 0
        Q_list[self.number_of_grid-1] = self.phi_L
        P_list[self.number_of_grid-1] = 0
        phi_list[0] = self.phi_0
        phi_list[self.number_of_grid-1] = self.phi_L

        # forward sweep to calculate P and Q at each node
        # please note that the calculation of a, b and c is implemented 
        # inside the loop as a more general code implementation. in this 
        # specific flow problem, since all fluid properties and flow velocities 
        # are constant across the domain, these coefficients actually constants 
        # and need to calculate once only outside the TDMA loop

        for i in range(1, self.number_of_grid-1):
            if self.scheme == "CDS":
                b = self.gamma/self.delta_x - self.rho*self.u/2
                c = self.gamma/self.delta_x + self.rho*self.u/2
                a = b + c

            elif self.scheme == "UDS":
                b = self.gamma/self.delta_x + max(-(self.rho*self.u), 0)
                c = self.gamma/self.delta_x + max(self.rho*self.u, 0)
                a = b + c

            elif self.scheme == "PLDS":
            
                b = self.gamma/self.delta_x * max(0, (1 - 0.1 * abs(self.Pe_local)) ** 5) + max(-(self.rho*self.u), 0)
                c = self.gamma/self.delta_x * max(0, (1 - 0.1 * abs(self.Pe_local)) ** 5) + max(self.rho*self.u, 0)
                a = b + c

            else:
                raise NotImplementedError
            
            P_list[i] = b / (a - c*P_list[i-1])
            Q_list[i] = c*Q_list[i-1] / (a - c*P_list[i-1])

        # backward sweep to calculate phi at each node
        for i in reversed(range(1, self.number_of_grid-1)):
            phi_list[i] = P_list[i]*phi_list[i+1] + Q_list[i]

        return phi_list



    def analytical_phi(self, x):
        
        phi_analytic = self.phi_0 + (((np.exp(self.rho*self.u*x/self.gamma))-1)/((np.exp(self.rho*self.u*self.L/self.gamma))-1)) *(self.phi_L - self.phi_0)

        return phi_analytic



    def error_calculation(self, phi_actual, phi_analytic):
        
        error = 100 * np.mean(np.abs((phi_actual - phi_analytic) / phi_analytic))
        
        return error


    def calculation_of_local_peclet_number(self):
        
        Pe_local = self.rho * self.u * self.delta_x / self.gamma

        return Pe_local




    # =================================================================
    # ========================= Visualisation =========================
    # =================================================================



    # set the font
    text_font = 'Arial'
    math_font = 'stix'
    plt.rcParams['mathtext.fontset'] = math_font
    rcParams['font.family'] = text_font


    def save_fig_custom(self, fig, file_path='', file_name='fig', format_list=['.eps', '.png'], overwrite=False, dpi = 500):
        
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



    def plot_the_result(self,  number_of_grid, u, scheme_list):
        
        self.u = u
        self.number_of_grid = number_of_grid
        x = np.linspace(0, self.L, self.number_of_grid)
        self.delta_x = self.L/(self.number_of_grid-1)       # end points are half-cell
        phi_analytic = self.analytical_phi(x)

        # plot the analytical solution smoothly
        x_for_analytic_plot = np.linspace(0, self.L, 999)
        phi_analytic_plot = self.analytical_phi(x_for_analytic_plot)

        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=plt.cm.Set1.colors)
        ax.plot(x_for_analytic_plot, phi_analytic_plot, label = 'analytic', ls = '--', zorder = 10)

        for scheme in scheme_list: 
            self.scheme = scheme
            self.Pe_local = self.calculation_of_local_peclet_number()
            start_time = time.time()
            phi_actual = self.solver_tdma()
            end_time = time.time()
            print(f"Computation time: {end_time - start_time} seconds for {scheme}.")
            error = self.error_calculation(phi_actual, phi_analytic)
            ax.plot(x, phi_actual, marker = 'o', label = f'{scheme} (error = {error:.2f}%)')
        ax.set_xlim(0,1)
        ax.set_ylim(15,105)
        ax.grid(ls = ':')
        ax.legend(loc = 'lower left', fontsize = 16)
        ax.set_xlabel('x (m)', fontsize = 16)
        ax.set_ylabel('$\phi$', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=16)

        self.save_fig_custom(fig, file_path='figures', file_name=f'phi_plot_grid_{self.number_of_grid}_u_{u}_pe_{self.Pe_local:.2f}', overwrite=True, dpi = 500)



    def error_plot(self, u_list, number_of_grid_list, scheme_list, mode):
        
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=plt.cm.Set1.colors[1:])
        
        for scheme in scheme_list: 
            self.scheme = scheme
            error_list = []
            delta_x_list = []
            
            if mode == 'grid_size':
                self.u = u_list[0]
                for number_of_grid in number_of_grid_list:
                    self.number_of_grid = number_of_grid
                    x = np.linspace(0, self.L, self.number_of_grid)
                    self.delta_x = self.L/(self.number_of_grid-1)
                    delta_x_list.append(self.delta_x)
                    phi_analytic = self.analytical_phi(x)
                    self.Pe_local = self.calculation_of_local_peclet_number()
                    phi_actual = self.solver_tdma()
                    error = self.error_calculation(phi_actual, phi_analytic)
                    error_list.append(error)
                ax.plot(delta_x_list, error_list, label = f'{self.scheme}')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Grid Size $\delta x$ (m)', fontsize = 16)

                def dx_to_pe(delta_x):
                    return self.rho * self.u * delta_x / self.gamma

                def pe_to_dx(pe):
                    return pe * self.gamma / (self.rho * self.u)

                secax = ax.secondary_xaxis('top', functions=(dx_to_pe, pe_to_dx))
                secax.set_xlabel('Local Péclet Number $Pe$', fontsize=16)
                secax.tick_params(axis='x', labelsize=16)

            elif mode == 'flow_velocity':
                self.number_of_grid = number_of_grid_list[0]
                for u in u_list:
                    self.u = u
                    x = np.linspace(0, self.L, self.number_of_grid)
                    self.delta_x = self.L/(self.number_of_grid-1)
                    phi_analytic = self.analytical_phi(x)
                    self.Pe_local = self.calculation_of_local_peclet_number()
                    phi_actual = self.solver_tdma()
                    error = self.error_calculation(phi_actual, phi_analytic)
                    error_list.append(error)
                ax.plot(u_list, error_list, label = f'{self.scheme}')
                ax.set_xlabel('Flow Velocity $u$ (m/s)', fontsize = 16)
                
                def u_to_pe(u):
                    return self.rho * u * self.delta_x / self.gamma

                def pe_to_u(pe):
                    return pe * self.gamma / (self.rho * self.delta_x)
                
                secax = ax.secondary_xaxis('top', functions=(u_to_pe, pe_to_u))
                secax.set_xlabel('Local Péclet Number $Pe$', fontsize=16)
                secax.tick_params(axis='x', labelsize=16)

            else:
                raise NotImplementedError
            
        ax.grid(ls = ':')
        ax.legend(loc = 'best', fontsize = 16)
        
        ax.set_ylabel('Percentage Mean Error', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        self.save_fig_custom(fig, file_path='figures', file_name=f'{mode}_error_plot', overwrite=True, dpi = 500)