#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  LID-DRIVEN CAVITY FLOW SOLVER
  Using the SIMPLE Algorithm on a Collocated FVM Grid
=============================================================================

PROBLEM DESCRIPTION:
    Solve incompressible Navier-Stokes equations in a unit square cavity
    where the top wall (y = L) moves at U_lid = 1 m/s.
    All other walls are stationary no-slip walls.

    Governing equations (steady, incompressible, Newtonian fluid):
        Continuity:   ∂u/∂x + ∂v/∂y = 0
        x-momentum:   ρ(u·∂u/∂x + v·∂u/∂y) = -∂p/∂x + μ(∂²u/∂x² + ∂²u/∂y²)
        y-momentum:   ρ(u·∂v/∂x + v·∂v/∂y) = -∂p/∂y + μ(∂²v/∂x² + ∂²v/∂y²)

NUMERICAL METHOD:
    - Finite Volume Method (FVM) on a uniform collocated Cartesian grid
    - SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm
    - Upwind Differencing Scheme (UDS / Power Law) for convection
    - Central Differencing for diffusion
    - Rhie-Chow interpolation to suppress pressure-velocity decoupling
      (checkerboard instability) on the collocated grid

GRID CONVENTION:
    u[i, j] where i = x-direction index, j = y-direction index
    Cell (i,j) has:
        East face  between (i,j) and (i+1,j)
        West face  between (i-1,j) and (i,j)
        North face between (i,j) and (i,j+1)
        South face between (i,j) and (i,j-1)

    y (j)
    ^
    |
    +---> x (i)

REFERENCES:
    - Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow.
    - Ferziger & Peric (2002). Computational Methods for Fluid Dynamics.
    - Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for
      incompressible flow using the Navier-Stokes equations and a multigrid
      method. Journal of Computational Physics, 48(3), 387-411.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_velocity_bcs(u, v, U_lid):
    """
    Apply Dirichlet (no-slip / moving lid) velocity boundary conditions.

    All walls enforce no-penetration and no-slip, except the top wall (j = -1,
    y = L) which is the moving lid.

    Note on indexing: u[i,j] → i is x, j is y
        u[0,:]  / u[-1,:] = left / right wall  (x = 0 or x = L)
        u[:,0]  / u[:,-1] = bottom / top wall  (y = 0 or y = L)
    """
    # Left wall (x = 0): no-slip
    u[0, :] = 0.0
    v[0, :] = 0.0

    # Right wall (x = L): no-slip
    u[-1, :] = 0.0
    v[-1, :] = 0.0

    # Bottom wall (y = 0): no-slip
    u[:, 0] = 0.0
    v[:, 0] = 0.0

    # Top wall (y = L): MOVING LID — drives the cavity flow
    u[:, -1] = U_lid   # horizontal velocity = U_lid
    v[:, -1] = 0.0     # no wall-normal penetration


# =============================================================================
# GRID AND PHYSICAL PARAMETERS
# =============================================================================

nx, ny = 61, 61         # Number of grid points in x and y
l = 1                   # Domain side length [m]

dx = l / (nx - 1)       # Grid spacing in x
dy = l / (ny - 1)       # Grid spacing in y

x = np.linspace(0, l, nx)
y = np.linspace(0, l, ny)

# Physical properties
nu  = 1e-2   # Kinematic viscosity [m²/s]  → Re = U*L/nu = 1*1/0.01 = 100
# nu = 2.5e-3  # Uncomment for Re = 400
rho = 1.0    # Density [kg/m³]
mu  = nu * rho  # Dynamic viscosity [Pa·s]

# Initial conditions: fluid at rest, zero pressure
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))

U_lid = 1.0  # Lid velocity [m/s]
apply_velocity_bcs(u, v, U_lid)


# =============================================================================
# SOLVER SETTINGS
# =============================================================================

# Under-relaxation factors (URF) — critical for SIMPLE stability:
#   Momentum URF: 0 = no update, 1 = full Newton step. Typically 0.3–0.7.
#   Pressure URF: much smaller (0.03–0.3) to avoid divergence.
urf_u = 0.5   # URF for u-momentum
urf_v = 0.5   # URF for v-momentum
urf_p = 0.2   # URF for pressure correction

n_iter  = 500  # Total SIMPLE outer iterations
gs_mom  = 15   # Gauss-Seidel sweeps per momentum solve
gs_p    = 50   # Gauss-Seidel sweeps per pressure-correction solve


# =============================================================================
# WORKING ARRAYS
# =============================================================================

# Central (diagonal) coefficients of the discretised momentum equations.
# Stored for every interior cell; also needed by Rhie-Chow interpolation.
au_P_arr = np.zeros_like(u)   # a_P for u-momentum equation
av_P_arr = np.zeros_like(v)   # a_P for v-momentum equation

# Coefficients for the pressure-correction (p') equation
a_E_prime = np.zeros_like(p)
a_W_prime = np.zeros_like(p)
a_N_prime = np.zeros_like(p)
a_S_prime = np.zeros_like(p)
a_P_prime = np.zeros_like(p)

p_prime = np.zeros_like(p)    # Pressure correction field p'
bP      = np.zeros_like(p)    # Mass imbalance (RHS of p'-equation)


# =============================================================================
# DIFFUSION COEFFICIENTS (constant for uniform grid)
# =============================================================================
# FVM diffusion coefficient = μ * (face area) / (distance between cell centres)
# For a uniform 2-D grid:
#   East/West faces have area dy (per unit depth), separation dx
#   North/South faces have area dx (per unit depth), separation dy
D_E = mu * dy / dx   # Diffusion coeff, East  face
D_W = mu * dy / dx   # Diffusion coeff, West  face
D_N = mu * dx / dy   # Diffusion coeff, North face
D_S = mu * dx / dy   # Diffusion coeff, South face


# =============================================================================
#   SIMPLE ALGORITHM — MAIN LOOP
# =============================================================================
#
#  Each outer iteration:
#    Step 1 — Solve u* from u-momentum using current pressure p
#    Step 2 — Solve v* from v-momentum using current pressure p
#    Step 3 — Compute mass imbalance via Rhie-Chow face velocities
#    Step 4 — Solve pressure-correction p' (Poisson equation)
#    Step 5 — Correct pressure:  p  ← p  + α_p * p'
#    Step 6 — Correct velocities: u ← u* − (dy/a_P) * ∂p'/∂x  (same for v)
#    Repeat until mass imbalance converges to zero.
#
# =============================================================================

for n in range(n_iter):
    print(f"Iteration {n}")

    # Save current velocities; u_star and v_star will be the predicted fields
    u_star = np.copy(u)
    v_star = np.copy(v)

    # Reset mass-imbalance array and pressure-correction guess each iteration
    bP[:]      = 0.0
    p_prime[:] = 0.0

    # =========================================================================
    # STEP 1: SOLVE U-MOMENTUM  →  get u*
    # =========================================================================
    #
    # Discretised u-momentum (power-law / upwind hybrid scheme):
    #
    #   a_P * u_P = Σ a_nb * u_nb  −  (p_E − p_W) * dy
    #
    # where the neighbour coefficients a_nb combine diffusion (D) and
    # convection (F) via the upwind scheme:
    #
    #   a_E = D_E + max(−F_E, 0)    ← upwind: only sees east if flow goes west
    #   a_W = D_W + max( F_W, 0)
    #   ... (same for N and S)
    #
    # The central coefficient enforces diagonal dominance:
    #   a_P = a_E + a_W + a_N + a_S + (F_E − F_W + F_N − F_S)
    #        ↑ sum of neighbours        ↑ net mass flux out of cell (≈0 if converged)
    #
    # Under-relaxation prevents overshooting:
    #   u_star_new = u_old + α_u * (u_new − u_old)
    #
    # =========================================================================

    for _ in range(gs_mom):
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                # --- Convective mass fluxes through each cell face [kg/s/m] ---
                # Face velocity interpolated as arithmetic mean of two neighbours
                # (linear interpolation on uniform grid).
                # Flux = ρ * (face normal velocity) * (face area)
                Fu_E = rho * dy * 0.5 * (u_star[i, j]   + u_star[i+1, j])  # East
                Fu_W = rho * dy * 0.5 * (u_star[i-1, j] + u_star[i, j])    # West
                Fu_N = rho * dx * 0.5 * (v_star[i, j]   + v_star[i, j+1])  # North
                Fu_S = rho * dx * 0.5 * (v_star[i, j-1] + v_star[i, j])    # South

                # --- Neighbour coefficients (upwind + diffusion) ---
                # max(−F, 0) adds the upwind contribution only when flow
                # is directed INTO the cell from that face.
                au_E = D_E + max(-Fu_E, 0)
                au_W = D_W + max( Fu_W, 0)
                au_N = D_N + max(-Fu_N, 0)
                au_S = D_S + max( Fu_S, 0)

                # --- Central coefficient ---
                # The extra term (Fu_E − Fu_W + Fu_N − Fu_S) is the net
                # outward mass flux; it's zero at convergence (continuity).
                # 1e-20 prevents division by zero before flow develops.
                au_P = (au_E + au_W + au_N + au_S
                        + (Fu_E - Fu_W + Fu_N - Fu_S)
                        + 1e-20)

                au_P_arr[i, j] = au_P   # store for later use in Rhie-Chow

                # --- RHS: neighbour contributions + pressure force ---
                # Pressure force on cell (i,j) in x-direction:
                #   F_p = −(p_e − p_w) * dy
                # With central differencing on a collocated grid:
                #   p_e − p_w ≈ (p[i+1] − p[i−1]) / 2
                # NOTE: This wide stencil is the collocated grid's weakness
                # (checkerboard modes). Rhie-Chow (Step 3) fixes continuity,
                # but the momentum equation still uses this central gradient.
                rhs = (
                      au_E * u_star[i+1, j]
                    + au_W * u_star[i-1, j]
                    + au_N * u_star[i, j+1]
                    + au_S * u_star[i, j-1]
                    - dy * (p[i+1, j] - p[i-1, j]) / 2   # pressure gradient * cell face area
                )

                u_new = rhs / au_P

                # Under-relaxation: blend new solution with old iterate
                u_star[i, j] = u[i, j] + urf_u * (u_new - u[i, j])

        # Enforce boundary conditions after each Gauss-Seidel sweep
        u_star[:, 0]  = 0.0     # bottom wall
        u_star[:, -1] = U_lid   # top lid
        u_star[0, :]  = 0.0     # left wall
        u_star[-1, :] = 0.0     # right wall


    # =========================================================================
    # STEP 2: SOLVE V-MOMENTUM  →  get v*
    # =========================================================================
    # Identical structure to u-momentum, but for the y-component of velocity.
    # Pressure gradient: −(p_N − p_S) * dx = −(p[i,j+1] − p[i,j−1])/2 * dx

    for _ in range(gs_mom):
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                # Convective fluxes (same faces, same interpolation)
                Fv_E = rho * dy * 0.5 * (u_star[i, j]   + u_star[i+1, j])
                Fv_W = rho * dy * 0.5 * (u_star[i-1, j] + u_star[i, j])
                Fv_N = rho * dx * 0.5 * (v_star[i, j]   + v_star[i, j+1])
                Fv_S = rho * dx * 0.5 * (v_star[i, j-1] + v_star[i, j])

                av_E = D_E + max(-Fv_E, 0)
                av_W = D_W + max( Fv_W, 0)
                av_N = D_N + max(-Fv_N, 0)
                av_S = D_S + max( Fv_S, 0)

                av_P = (av_E + av_W + av_N + av_S
                        + (Fv_E - Fv_W + Fv_N - Fv_S)
                        + 1e-20)

                av_P_arr[i, j] = av_P

                rhs = (
                      av_E * v_star[i+1, j]
                    + av_W * v_star[i-1, j]
                    + av_N * v_star[i, j+1]
                    + av_S * v_star[i, j-1]
                    - dx * (p[i, j+1] - p[i, j-1]) / 2   # y-direction pressure gradient
                )

                v_new = rhs / av_P

                v_star[i, j] = v[i, j] + urf_v * (v_new - v[i, j])

        # Enforce boundary conditions after each Gauss-Seidel sweep
        v_star[:, 0]  = 0.0   # bottom wall
        v_star[:, -1] = 0.0   # top wall (lid has zero normal velocity)
        v_star[0, :]  = 0.0   # left wall
        v_star[-1, :] = 0.0   # right wall


    # =========================================================================
    # EXTRAPOLATE a_P ARRAYS TO BOUNDARY CELLS
    # =========================================================================
    # au_P_arr and av_P_arr were only filled for interior cells (range 1..n-2).
    # Boundary cells remain 0 from initialisation. The Rhie-Chow interpolation
    # below computes 1/a_P for faces adjacent to walls — without this step,
    # those faces would produce 1/0 = ∞, immediately causing NaN.
    # Simple first-order extrapolation (copy nearest interior value):

    au_P_arr[0, :]  = au_P_arr[1, :]    # left ghost row
    au_P_arr[-1, :] = au_P_arr[-2, :]   # right ghost row
    au_P_arr[:, 0]  = au_P_arr[:, 1]    # bottom ghost column
    au_P_arr[:, -1] = au_P_arr[:, -2]   # top ghost column

    av_P_arr[0, :]  = av_P_arr[1, :]
    av_P_arr[-1, :] = av_P_arr[-2, :]
    av_P_arr[:, 0]  = av_P_arr[:, 1]
    av_P_arr[:, -1] = av_P_arr[:, -2]


    # =========================================================================
    # STEP 3: COMPUTE MASS IMBALANCE (Rhie-Chow face velocities)
    # =========================================================================
    #
    # WHY RHIE-CHOW?
    # On a collocated grid, interpolating u* directly to faces and computing
    # the divergence couples only alternating cells — the classic checkerboard
    # pressure instability. Rhie & Chow (1983) proposed correcting the face
    # velocity by adding a pressure-smoothing term:
    #
    #   u_e = avg(u*_P, u*_E) − avg(dy/a_P) * (p_E − p_P)
    #                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                            compact face pressure gradient
    #                            (uses ADJACENT cell centres, not i±1)
    #
    # This ensures that even/odd pressure modes do not satisfy continuity,
    # so they are driven out by the pressure-correction step.
    #
    # The coefficient avg(dy/a_P) = 0.5*(dy/a_P_P + dy/a_P_E)
    # This is the correct Rhie-Chow form; note it differs from
    # dy/(a_P_P + a_P_E) by a factor of ~2 when a_P values are equal.
    #
    # bP[i,j] = net outward mass flux = continuity residual for cell (i,j)
    # At convergence bP → 0 everywhere.
    #
    # =========================================================================

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            # --- East face velocity (u_e) ---
            u_E = ( 0.5 * (u_star[i, j] + u_star[i+1, j])
                  - 0.5 * dy * (1.0/au_P_arr[i, j] + 1.0/au_P_arr[i+1, j])
                  * (p[i+1, j] - p[i, j]) )
            F_E = rho * u_E * dy

            # --- West face velocity (u_w) ---
            u_W = ( 0.5 * (u_star[i-1, j] + u_star[i, j])
                  - 0.5 * dy * (1.0/au_P_arr[i-1, j] + 1.0/au_P_arr[i, j])
                  * (p[i, j] - p[i-1, j]) )
            F_W = rho * u_W * dy

            # --- North face velocity (v_n) ---
            v_N = ( 0.5 * (v_star[i, j] + v_star[i, j+1])
                  - 0.5 * dx * (1.0/av_P_arr[i, j] + 1.0/av_P_arr[i, j+1])
                  * (p[i, j+1] - p[i, j]) )
            F_N = rho * v_N * dx

            # --- South face velocity (v_s) ---
            v_S = ( 0.5 * (v_star[i, j-1] + v_star[i, j])
                  - 0.5 * dx * (1.0/av_P_arr[i, j-1] + 1.0/av_P_arr[i, j])
                  * (p[i, j] - p[i, j-1]) )
            F_S = rho * v_S * dx

            # Net outward mass flux = continuity residual
            bP[i, j] = F_E - F_W + F_N - F_S

    print(f"  Max mass imbalance: {np.max(np.abs(bP)):.6e}")


    # =========================================================================
    # STEP 4: PRESSURE-CORRECTION EQUATION
    # =========================================================================
    #
    # The idea: if we apply a pressure correction p' such that
    #   p_new = p + p'
    # and simultaneously correct the velocities:
    #   u_new = u* − (dy/a_P) * (p'_E − p'_P)   (compact gradient)
    #
    # then substituting into continuity gives a POISSON equation for p':
    #
    #   a_E'*p'_E + a_W'*p'_W + a_N'*p'_N + a_S'*p'_S − a_P'*p'_P = bP
    #
    # where the coefficients are derived from the Rhie-Chow velocity expressions:
    #
    #   a_E'[i,j] = ρ * dy² * 0.5 * (1/a_P[i,j] + 1/a_P[i+1,j])
    #
    # This is exactly consistent with the Rhie-Chow face velocities computed
    # in Step 3 — consistency is essential for correct mass conservation.
    # =========================================================================

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            # East: driven by u-face between (i,j) and (i+1,j)
            a_E_prime[i, j] = (rho * dy**2 * 0.5
                               * (1.0/(au_P_arr[i, j]   + 1e-20)
                                + 1.0/(au_P_arr[i+1, j] + 1e-20)))

            # West: driven by u-face between (i-1,j) and (i,j)
            a_W_prime[i, j] = (rho * dy**2 * 0.5
                               * (1.0/(au_P_arr[i-1, j] + 1e-20)
                                + 1.0/(au_P_arr[i, j]   + 1e-20)))

            # North: driven by v-face between (i,j) and (i,j+1)
            a_N_prime[i, j] = (rho * dx**2 * 0.5
                               * (1.0/(av_P_arr[i, j]   + 1e-20)
                                + 1.0/(av_P_arr[i, j+1] + 1e-20)))

            # South: driven by v-face between (i,j-1) and (i,j)
            a_S_prime[i, j] = (rho * dx**2 * 0.5
                               * (1.0/(av_P_arr[i, j-1] + 1e-20)
                                + 1.0/(av_P_arr[i, j]   + 1e-20)))

            a_P_prime[i, j] = (a_E_prime[i, j] + a_W_prime[i, j]
                              + a_N_prime[i, j] + a_S_prime[i, j]
                              + 1e-20)

    # --- Gauss-Seidel iterations for p' ---
    # Starting from p_prime = 0 each outer iteration (fresh start).
    for _ in range(gs_p):
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                p_prime[i, j] = (
                      a_E_prime[i, j] * p_prime[i+1, j]
                    + a_W_prime[i, j] * p_prime[i-1, j]
                    + a_N_prime[i, j] * p_prime[i, j+1]
                    + a_S_prime[i, j] * p_prime[i, j-1]
                    - bP[i, j]
                ) / a_P_prime[i, j]

        # Neumann (zero-gradient) BCs for p':
        # The pressure-correction equation has no Dirichlet BCs on walls
        # (all walls are impermeable), so dp'/dn = 0 everywhere on the boundary.
        p_prime[0, :]  = p_prime[1, :]    # left wall:   dp'/dx = 0
        p_prime[-1, :] = p_prime[-2, :]   # right wall:  dp'/dx = 0
        p_prime[:, 0]  = p_prime[:, 1]    # bottom wall: dp'/dy = 0
        p_prime[:, -1] = p_prime[:, -2]   # top wall:    dp'/dy = 0

    # Remove the mean of p' to fix the pressure level (p is only defined up
    # to an additive constant for enclosed flows with no pressure BCs).
    p_prime -= np.mean(p_prime)


    # =========================================================================
    # STEP 5: CORRECT PRESSURE
    # =========================================================================
    # Under-relaxation on pressure avoids large oscillations:
    #   p ← p + α_p * p'
    # α_p is typically much smaller than momentum URFs (e.g. 0.03–0.3).

    p = p + urf_p * p_prime


    # =========================================================================
    # STEP 6: CORRECT VELOCITIES
    # =========================================================================
    #
    # Derived from the momentum equation after subtracting the u* equation:
    #   a_P * (u − u*) = −dy * (p'_E − p'_P)
    #   ⟹  u = u* − (dy / a_P) * (p'_E − p'_P)
    #
    # Uses a COMPACT gradient (p'[i+1,j] − p'[i,j]), consistent with
    # the Rhie-Chow interpolation and p'-equation derivation.
    # Note: NO under-relaxation on velocity here — it was already applied
    # during the momentum solve (u_star already includes URF).
    #
    # =========================================================================

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i, j] = u_star[i, j] - (dy / au_P_arr[i, j]) * (p_prime[i+1, j] - p_prime[i, j])
            v[i, j] = v_star[i, j] - (dx / av_P_arr[i, j]) * (p_prime[i, j+1] - p_prime[i, j])

    # Re-enforce boundary conditions on the corrected velocity field
    apply_velocity_bcs(u, v, U_lid)


# =============================================================================
# POST-PROCESSING AND VISUALISATION
# =============================================================================

X, Y = np.meshgrid(x, y, indexing='ij')

# --- Pressure and velocity quiver plot ---
fig = plt.figure(figsize=(11, 7), dpi=100)
cf = plt.contourf(X, Y, p, alpha=0.5, cmap='turbo', levels=20)
plt.colorbar(cf, label='Pressure')
contour = plt.contour(X, Y, p, cmap='turbo', levels=10)
plt.clabel(contour, inline=False, fontsize=12, colors='black')
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Pressure Contours and Velocity Field', fontsize=14)
plt.show()

# --- Streamlines ---
plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(x, y, p.T, levels=50, cmap='coolwarm')
plt.colorbar()
plt.streamplot(x, y, u.T, v.T, density=1.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.title('Streamlines', fontsize=14)
plt.show()

# Re-create coordinate arrays after they were overwritten above
y = np.linspace(0, 1, ny)
x = np.linspace(0, 1, nx)

# =============================================================================
# GHIA et al. (1982) BENCHMARK DATA — Re = 100
# =============================================================================
# Ghia solved the same problem with a multigrid vorticity-streamfunction
# approach on a 129×129 grid. Their data are the standard validation target.

y_ghia = np.array([1.0000, 0.9766, 0.8594, 0.7344, 0.6172,
                   0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0000])
u_ghia = np.array([1.0000,  0.8412,  0.3330,  0.0626, -0.0608,
                  -0.2109, -0.2058, -0.1566, -0.1034, -0.0643,  0.0000])

x_ghia = np.array([0.0000, 0.0625, 0.0938, 0.1406, 0.5000, 0.7734, 0.9063, 1.0000])
v_ghia = np.array([0.0000, 0.0923, 0.1009, 0.1065, 0.0000, -0.0853, -0.0982, 0.0000])

# =============================================================================
# GHIA et al. (1982) BENCHMARK DATA — Re = 400
# =============================================================================
y_ghia_400 = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
                       0.8516, 0.7344, 0.6172, 0.5000,
                       0.4531, 0.2813, 0.1719, 0.1016, 0.0000])
u_ghia_400 = np.array([ 1.0000,  0.7584,  0.6844,  0.6176,  0.5589,
                        0.2903,  0.1626,  0.0214, -0.1364,
                       -0.1712, -0.1148, -0.0605, -0.0320,  0.0000])

x_ghia_400 = np.array([0.0000, 0.0313, 0.0625, 0.0938, 0.1250,
                       0.5000, 0.7656, 0.7969, 0.8281,
                       0.8594, 0.8906, 0.9219, 0.9531, 1.0000])
v_ghia_400 = np.array([ 0.0000,  0.0719,  0.1189,  0.1500,  0.1684,
                        0.0000, -0.1608, -0.1732, -0.1811,
                       -0.1840, -0.1824, -0.1762, -0.1649,  0.0000])


# Centreline slices from the computed solution
u_mid = u[nx//2, :]    # u at x = 0.5 (varying y)
v_mid = v[:, ny//2]    # v at y = 0.5 (varying x)

# --- u-velocity profile at x = 0.5 ---
plt.figure()
plt.plot(u_mid, y, label="My SIMPLE solver")
plt.scatter(u_ghia, y_ghia, color="k", marker="o", label="Ghia et al. (1982) Re=100")
plt.xlabel("u")
plt.ylabel("y")
plt.title("Centreline u-velocity (x = 0.5)")
plt.legend()
plt.grid()
plt.show()

# --- v-velocity profile at y = 0.5 ---
plt.figure()
plt.plot(x, v_mid, label="My SIMPLE solver")
plt.scatter(x_ghia, v_ghia, color="k", marker="o", label="Ghia et al. (1982) Re=100")
plt.xlabel("x")
plt.ylabel("v")
plt.title("Centreline v-velocity (y = 0.5)")
plt.legend()
plt.grid()
plt.show()