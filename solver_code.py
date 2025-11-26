import numpy as np
import matplotlib.pyplot as plt





# Physical parameters setting
L = 1.0             # domain length
rho = 1.0           # fluid density
Gamma = 0.1         # fluid diffusion coefficient
phi_0 = 100.0       # left boundary value
phi_L = 20.0        # right boundary value





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


def analytical_phi(x, L, rho, u, Gamma, phi_0, phi_L):
    Pe = rho * u * L / Gamma  # Péclet number
    
    # if abs(u) < 1e-14:
    #     # pure diffusion -> linear profile
    #     return phi_0 + (phi_L - phi_0) * x / L
    
    # if abs(Pe) < 1e-6:
    #     # limit Pe -> 0 also gives linear
    #     return phi_0 + (phi_L - phi_0) * x / L

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
        if scheme == "CD":             # Central Differencing
            a_W = D + 0.5 * F
            a_E = D - 0.5 * F

        elif scheme == "UD":           # Upwind Differencing
            if F >= 0.0:
                a_W = D + F
                a_E = D
            else:
                # for negative flow direction (right-to-left)
                a_W = D
                a_E = D - F

        elif scheme == "PL":           # Power-law Differencing (Patankar)
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



# if __name__ == "__main__":

#     schemes = {
#         "CD": "Central differencing",
#         "UD": "Upwind differencing",
#         "PL": "Power-law differencing",
#     }

    # # -------------------------------------------------
    # # 1. Profiles for different velocities (fixed grid)
    # # -------------------------------------------------
    # Nx = Nx_for_profiles

    # for scheme_id, scheme_name in schemes.items():
    #     plt.figure()
    #     for u in velocities:
    #         x, phi_num = solver(
    #             Nx, L, rho, u, Gamma, phi_0, phi_L, scheme_id
    #         )
    #         phi_exact = analytical_phi(x, L, rho, u, Gamma, phi_0, phi_L)
    #         plt.plot(x, phi_num, marker="o", linestyle="-", label=f"u = {u} m/s (num)")
    #         plt.plot(x, phi_exact, linestyle="--", label=f"u = {u} m/s (analytic)")

    #     plt.xlabel("x [m]")
    #     plt.ylabel(r"$\phi$")
    #     plt.title(f"Convection–diffusion profiles ({scheme_name}), Nx = {Nx}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()

    # # -------------------------------------------------
    # # 2. Error vs grid spacing for each scheme (fixed u)
    # # -------------------------------------------------
    # plt.figure()
    # for scheme_id, scheme_name in schemes.items():
    #     dx_list = []
    #     err_list = []
    #     for Nx in Nx_list_error:
    #         x, phi_num = solver(
    #             Nx, L, rho, u_for_error_plot, Gamma, phi_0, phi_L, scheme_id
    #         )
    #         phi_exact = analytical_phi(x, L, rho, u_for_error_plot, Gamma, phi_0, phi_L)
    #         # use internal nodes only in the error, consistent with discretisation
    #         err = mean_absolute_percentage_error(phi_num[1:-1], phi_exact[1:-1])
    #         dx = L / (Nx - 1)
    #         dx_list.append(dx)
    #         err_list.append(err)

    #     plt.loglog(dx_list, err_list, marker="o", linestyle="-", label=scheme_name)

    # plt.xlabel(r"Grid spacing $\Delta x$ [m]")
    # plt.ylabel("Mean absolute percentage error [%]")
    # plt.title(f"Error vs grid spacing, u = {u_for_error_plot} m/s")
    # plt.legend()
    # plt.grid(True, which="both")
    # plt.tight_layout()

    # plt.show()
