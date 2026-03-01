#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 18:16:22 2026

@author: akim
"""

import numpy as np
import matplotlib.pyplot as plt


def apply_velocity_bcs(u, v, U_lid):

    u[0, :] = 0.0
    v[0, :] = 0.0

    u[-1, :] = 0.0
    v[-1, :] = 0.0

    u[:, 0] = 0.0
    v[:, 0] = 0.0

    u[:, -1] = U_lid
    v[:, -1] = 0.0

# -----------------------------
# Grid, parameters, initial & BC
# -----------------------------
nx, ny = 61, 61
l = 1

dx = l / (nx - 1)
dy = l / (ny - 1)

x = np.linspace(0, l, nx)
y = np.linspace(0, l, ny)
nu = 1e-2 #kinematic viscosity
# nu = 2.5e-3 # Re = 400
rho = 1
mu = nu*rho

u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))

# Lid-driven cavity BCs
U_lid = 1
apply_velocity_bcs(u, v, U_lid)

# Under-relaxation
urf_u = 0.5
urf_v = 0.5
urf_p = 0.2

n_iter = 500
gs_mom = 15
gs_p = 50

au_P_arr = np.zeros_like(u)
av_P_arr = np.zeros_like(v)
a_E_prime = np.zeros_like(p)
a_W_prime = np.zeros_like(p)
a_N_prime = np.zeros_like(p)
a_S_prime = np.zeros_like(p)
a_P_prime = np.zeros_like(p)
p_prime = np.zeros_like(p)
bP = np.zeros_like(p)

# ---- Diffusion coefficients ----
D_E = mu * dy / dx
D_W = mu * dy / dx
D_N = mu * dx / dy
D_S = mu * dx / dy



# SIMPLE LOOP
for n in range(n_iter):
    print(n)
    u_star = np.copy(u)
    v_star = np.copy(v)
    bP[:] = 0.0
    p_prime[:] = 0.0   # initial guess
    # Gauss-Seidel loop to calculate predicted velocities
    # x-mom
    for _ in range(gs_mom):
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                # ----- Fluxes through faces ----
                # u-momentum fluxes
                Fu_E = rho * dy * 0.5 * (u_star[i,j] + u_star[i+1,j])
                Fu_W = rho * dy * 0.5 * (u_star[i-1,j] + u_star[i,j])
                Fu_N = rho * dx * 0.5 * (v_star[i,j] + v_star[i,j+1])
                Fu_S = rho * dx * 0.5 * (v_star[i,j-1] + v_star[i,j])

                au_E = D_E + max(-Fu_E,0)
                au_W = D_W + max(Fu_W,0)
                au_N = D_N + max(-Fu_N,0)
                au_S = D_S + max(Fu_S,0)

                au_P = au_E + au_W + au_N + au_S + (Fu_E - Fu_W + Fu_N - Fu_S) + 1e-20 # +1e-20 to avoid 0 division

                au_P_arr[i,j] = au_P

                # RHS
                rhs = (
                    au_E * u_star[i+1,j] +
                    au_W * u_star[i-1,j] +
                    au_N * u_star[i,j+1] +
                    au_S * u_star[i,j-1] -
                    dy * (p[i+1,j] - p[i-1,j]) / 2
                )

                u_new = rhs / au_P

                u_star[i,j] = u[i,j] + urf_u * (u_new - u[i,j])

        # BC
        u_star[:, 0]  = 0.0
        u_star[:, -1] = U_lid
        u_star[0, :]  = 0.0
        u_star[-1, :] = 0.0

    # y-mom
    for _ in range(gs_mom):
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                # v-momentum fluxes
                Fv_E = rho * dy * 0.5 * (u_star[i,j] + u_star[i+1,j])
                Fv_W = rho * dy * 0.5 * (u_star[i-1,j] + u_star[i,j])
                Fv_N = rho * dx * 0.5 * (v_star[i,j] + v_star[i,j+1])
                Fv_S = rho * dx * 0.5 * (v_star[i,j-1] + v_star[i,j])

                av_E = D_E + max(-Fv_E,0)
                av_W = D_W + max(Fv_W,0)
                av_N = D_N + max(-Fv_N,0)
                av_S = D_S + max(Fv_S,0)

                av_P = av_E + av_W + av_N + av_S + (Fv_E - Fv_W + Fv_N - Fv_S) + 1e-20

                av_P_arr[i, j] = av_P

                 # ---- RHS ----
                rhs = (
                    av_E * v_star[i+1, j] +
                    av_W * v_star[i-1, j] +
                    av_N * v_star[i, j+1] +
                    av_S * v_star[i, j-1] -
                    dx * (p[i, j+1] - p[i, j-1]) / 2
                )

                v_new = rhs / av_P

                # ---- UNDER-RELAXATION ----
                v_star[i, j] = v[i, j] + urf_v * (v_new - v[i, j])

        # ---- Boundary conditions after each sweep ----
        v_star[:, 0]  = 0.0
        v_star[:, -1] = 0.0
        v_star[0, :]  = 0.0
        v_star[-1, :] = 0.0
        
    au_P_arr[0, :]  = au_P_arr[1, :]
    au_P_arr[-1, :] = au_P_arr[-2, :]
    au_P_arr[:, 0]  = au_P_arr[:, 1]
    au_P_arr[:, -1] = au_P_arr[:, -2]
    
    av_P_arr[0, :]  = av_P_arr[1, :]
    av_P_arr[-1, :] = av_P_arr[-2, :]
    av_P_arr[:, 0]  = av_P_arr[:, 1]
    av_P_arr[:, -1] = av_P_arr[:, -2]

    # Compute mass imbalance - Improved Rhie-Chow (consistent & stable)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # East face
            u_E = 0.5*(u_star[i,j] + u_star[i+1,j]) \
                  - 0.5 * dy * (1.0/au_P_arr[i,j] + 1.0/au_P_arr[i+1,j]) * (p[i+1,j] - p[i,j])
            F_E = rho * u_E * dy
            # West face
            u_W = 0.5*(u_star[i-1,j] + u_star[i,j]) \
                  - 0.5 * dy * (1.0/au_P_arr[i-1,j] + 1.0/au_P_arr[i,j]) * (p[i,j] - p[i-1,j])
            F_W = rho * u_W * dy
            # North face
            v_N = 0.5*(v_star[i,j] + v_star[i,j+1]) \
                  - 0.5 * dx * (1.0/av_P_arr[i,j] + 1.0/av_P_arr[i,j+1]) * (p[i,j+1] - p[i,j])
            F_N = rho * v_N * dx
            # South face
            v_S = 0.5*(v_star[i,j-1] + v_star[i,j]) \
                  - 0.5 * dx * (1.0/av_P_arr[i,j-1] + 1.0/av_P_arr[i,j]) * (p[i,j] - p[i,j-1])
            F_S = rho * v_S * dx
            
            bP[i,j] = F_E - F_W + F_N - F_S
            
    # print(f'Mass imbalance: {sum(sum(bP))}')
    print(f'Mass imbalance: {np.max(np.abs(bP))}')
    
    # Pressure-correction coefficients - Improved Rhie-Chow (standard & stable)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            a_E_prime[i,j] = rho * dy * dy * 0.5 * \
                (1.0/(au_P_arr[i,j] + 1e-20) + 1.0/(au_P_arr[i+1,j] + 1e-20))
            a_W_prime[i,j] = rho * dy * dy * 0.5 * \
                (1.0/(au_P_arr[i-1,j] + 1e-20) + 1.0/(au_P_arr[i,j] + 1e-20))
            a_N_prime[i,j] = rho * dx * dx * 0.5 * \
                (1.0/(av_P_arr[i,j] + 1e-20) + 1.0/(av_P_arr[i,j+1] + 1e-20))
            a_S_prime[i,j] = rho * dx * dx * 0.5 * \
                (1.0/(av_P_arr[i,j-1] + 1e-20) + 1.0/(av_P_arr[i,j] + 1e-20))
            a_P_prime[i,j] = (
                a_E_prime[i,j] + a_W_prime[i,j] +
                a_N_prime[i,j] + a_S_prime[i,j] + 1e-20
            )


    # Gauss-Seidel loop to calculate pressure correction
    for _ in range(gs_p):
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                p_prime[i,j] = (
                    a_E_prime[i,j] * p_prime[i+1,j]
                  + a_W_prime[i,j] * p_prime[i-1,j]
                  + a_N_prime[i,j] * p_prime[i,j+1]
                  + a_S_prime[i,j] * p_prime[i,j-1]
                  - bP[i,j]
                ) / a_P_prime[i,j]

        # --- Pressure-correction BCs (Neumann) ---
        p_prime[0, :]  = p_prime[1, :]
        p_prime[-1, :] = p_prime[-2, :]
        p_prime[:, 0]  = p_prime[:, 1]
        p_prime[:, -1] = p_prime[:, -2]

    p_prime -= np.mean(p_prime)

    # corrrected pressure
    p = p + urf_p * p_prime

    # corrected velocities
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u[i,j] = u_star[i,j] - (dy / au_P_arr[i,j]) * (p_prime[i+1,j] - p_prime[i,j])
            v[i,j] = v_star[i,j] - (dx / av_P_arr[i,j]) * (p_prime[i,j+1] - p_prime[i,j])
    apply_velocity_bcs(u, v, U_lid)


X, Y = np.meshgrid(x, y, indexing='ij')

fig = plt.figure(figsize=(11,7), dpi=100)
cf = plt.contourf(X, Y, p, alpha=0.5, cmap='turbo', levels=20)
plt.colorbar(cf, label='Pressure')
contour = plt.contour(X, Y, p, cmap='turbo', levels=10)
plt.clabel(contour, inline=False, fontsize=12, colors = 'black')
quiv = plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Pressure and Velocity fields', fontsize=14)
plt.show()

plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(x, y, p.T, levels=50, cmap='coolwarm')
plt.colorbar()
plt.streamplot(x, y, u.T, v.T, density=1.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()

y = np.linspace(0, 1, ny)
x = np.linspace(0, 1, nx)
# -----------------------------
# Ghia Re = 100
# -----------------------------
y_ghia = np.array([
    1.0000, 0.9766, 0.8594, 0.7344, 0.6172,
    0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0000
])

u_ghia = np.array([
    1.0000, 0.8412, 0.3330, 0.0626, -0.0608,
   -0.2109, -0.2058, -0.1566, -0.1034, -0.0643, 0.0000
])

x_ghia = np.array([
    0.0000, 0.0625, 0.0938, 0.1406, 0.5000,
    0.7734, 0.9063, 1.0000
])

v_ghia = np.array([
    0.0000, 0.0923, 0.1009, 0.1065, 0.0000,
   -0.0853, -0.0982, 0.0000
])

# -----------------------------
# Ghia Re = 400
# -----------------------------

y_ghia_400 = np.array([
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
    0.8516, 0.7344, 0.6172, 0.5000,
    0.4531, 0.2813, 0.1719, 0.1016, 0.0000
])

u_ghia_400 = np.array([
    1.0000, 0.7584, 0.6844, 0.6176, 0.5589,
    0.2903, 0.1626, 0.0214, -0.1364,
   -0.1712, -0.1148, -0.0605, -0.0320, 0.0000
])

x_ghia_400 = np.array([
    0.0000, 0.0313, 0.0625, 0.0938, 0.1250,
    0.5000, 0.7656, 0.7969, 0.8281,
    0.8594, 0.8906, 0.9219, 0.9531, 1.0000
])

v_ghia_400 = np.array([
    0.0000, 0.0719, 0.1189, 0.1500, 0.1684,
    0.0000, -0.1608, -0.1732, -0.1811,
   -0.1840, -0.1824, -0.1762, -0.1649, 0.0000
])



y = np.linspace(0, 1, ny)
x = np.linspace(0, 1, nx)

u_mid = u[nx//2, :]
v_mid = v[:, ny//2]

# --- u velocity at x = 0.5 ---
plt.figure()
plt.plot(u_mid, y, label="My SIMPLE solver")
plt.scatter(u_ghia, y_ghia, color="k", marker="o", label="Ghia et al. (1982)")
plt.xlabel("u")
plt.ylabel("y")
plt.title("Centerline u-velocity (x = 0.5)")
plt.legend()
plt.grid()
plt.show()

# --- v velocity at y = 0.5 ---
plt.figure()
plt.plot(x, v_mid, label="My SIMPLE solver")
plt.scatter(x_ghia, v_ghia, color="k", marker="o", label="Ghia et al. (1982)")
plt.xlabel("x")
plt.ylabel("v")
plt.title("Centerline v-velocity (y = 0.5)")
plt.legend()
plt.grid()
plt.show()
