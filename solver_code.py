import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os



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







def tdma(a_sub, a_diag, a_sup, d):
    """
    Solve a tridiagonal system A x = d using TDMA.
    a_sub: sub-diagonal (length n-1)
    a_diag: diagonal (length n)
    a_sup: super-diagonal (length n-1)
    d: RHS (length n)
    """
    n = len(a_diag)
    ac, bc, cc, dc = map(np.array, (a_sub, a_diag, a_sup, d))

    # special case: single equation
    if n == 1:
        return np.array([dc[0] / bc[0]])

    # forward sweep
    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - m * cc[i - 1]
        dc[i] = dc[i] - m * dc[i - 1]

    # back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x


def tdma(phi_0, phi_N, rho, u, gamma, L, number_of_grid):
    
    delta_x = L/(number_of_grid-1)      # first and last grid is half-cell, so full cell number is N-1
    
    # initialise the P, Q and phi lists
    Q_list = np.zeros(number_of_grid)
    P_list = np.zeros(number_of_grid)
    phi_list = np.zeros(number_of_grid)
    
    
    a = 2*gamma/delta_x
    b = gamma/delta_x - rho*u/2
    c = gamma/delta_x + rho*u/2
    
    Q_list[0] = phi_0
    P_list[0] = 0
    Q_list[number_of_grid-1] = phi_N
    P_list[number_of_grid] = 0
    phi_list[0] = phi_0
    phi_list[number_of_grid-1] = phi_N

    for i in range(1, number_of_grid-1):
        P_list[i] = b / (a - c*P_list[i-1])
        Q_list[i] = c*Q_list[i-1] / (a - c*P_list[i-1])

    for i in reversed(range(1, number_of_grid-1)):
        phi_list[i] = P_list[i]*phi_list[i+1] + Q_list[i]

    return phi_list






def analytical_phi(x, L, rho, u, Gamma, phi_0, phi_L):
    Pe = rho * u * L / Gamma  # Péclet number

    phi = phi_0 + (((np.exp(rho*u*x/Gamma))-1)/((np.exp(rho*u*L/Gamma))-1)) *(phi_L - phi_0)

    return phi, Pe



def solver(Nx, L, rho, u, Gamma, phi_0, phi_L, scheme):

    if Nx < 3:
        raise ValueError("Nx must be at least 3 (including boundary points).")

    # initialise the descretised domain
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]

    # constants
    D = Gamma / dx               # diffusion conductance
    F = rho * u                  # convection flux (assumed constant)

    # number of internal nodes (unknowns)
    n_int = Nx - 2

    # tri-diagonal arrays for internal nodes only
    a_sub = np.zeros(n_int - 1)   # sub-diagonal
    a_diag = np.zeros(n_int)      # diagonal
    a_sup = np.zeros(n_int - 1)   # super-diagonal
    rhs = np.zeros(n_int)         # right-hand side

    # loop over internal nodes i = 1..Nx-2
    for i in range(1, Nx - 1):
        k = i - 1  # internal index 0..n_int-1

        # convection-diffusion coefficients at node i
        if scheme == "CDS":             # Central Differencing
            a_W = D + 0.5 * F
            a_E = D - 0.5 * F

        elif scheme == "UDS":           # Upwind Differencing
            if F >= 0.0:
                a_W = D + F
                a_E = D
            else:
                # for negative flow direction (right-to-left)
                a_W = D
                a_E = D - F

        elif scheme == "PLDS":           # Power-law Differencing
            Pe = F / D
            phi_factor = max(0.0, (1.0 - 0.1 * abs(Pe)) ** 5)
            if F >= 0.0:
                a_W = D * phi_factor + F
                a_E = D * phi_factor
            else:
                a_W = D * phi_factor
                a_E = D * phi_factor - F
        else:
            raise NotImplementedError

        A_P = a_W + a_E  # no source term


        a_diag[k] = A_P

        # neighbours and boundary contributions
        # left boundary at i = 0, right boundary at i = Nx-1
        if i == 1:
            # left neighbour is boundary (x = 0)
            a_sup[k] = -a_E
            rhs[k] += a_W * phi_0
        elif i == Nx - 2:
            # right neighbour is boundary (x = L)
            a_sub[k - 1] = -a_W
            rhs[k] += a_E * phi_L
        else:
            # both neighbours are internal
            a_sub[k - 1] = -a_W
            a_sup[k] = -a_E

    # solve tri-diagonal system
    phi_internal = tdma(a_sub, a_diag, a_sup, rhs)

    # assemble full solution including boundaries
    phi = np.zeros(Nx)
    phi[0] = phi_0
    phi[-1] = phi_L
    phi[1:-1] = phi_internal

    return x, phi


def mean_absolute_percentage_error(phi_num, phi_exact):

    return 100.0 * np.mean(np.abs((phi_num - phi_exact) / phi_exact))