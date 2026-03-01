# Lid-Driven Cavity Flow Solver — SIMPLE Algorithm

A Python implementation of the **SIMPLE** (Semi-Implicit Method for Pressure-Linked Equations) algorithm to solve the incompressible Navier-Stokes equations in a 2D lid-driven cavity using the Finite Volume Method (FVM).

---

## Problem Description

A unit square cavity is filled with an incompressible Newtonian fluid. The **top wall moves horizontally** at a constant velocity `U_lid = 1 m/s`, driving the flow, while all other walls are stationary no-slip boundaries.

The steady-state governing equations are:

- **Continuity:** ∂u/∂x + ∂v/∂y = 0  
- **x-momentum:** ρ(u·∂u/∂x + v·∂u/∂y) = −∂p/∂x + μ∇²u  
- **y-momentum:** ρ(u·∂v/∂x + v·∂v/∂y) = −∂p/∂y + μ∇²v  

---

## Numerical Method

| Component | Method |
|---|---|
| Discretisation | Finite Volume Method (FVM), uniform collocated Cartesian grid |
| Algorithm | SIMPLE (Patankar, 1980) |
| Convection scheme | Upwind Differencing / Power-Law |
| Diffusion | Central Differencing |
| Pressure-velocity coupling | Rhie-Chow interpolation (prevents checkerboard instability) |
| Linear solver | Gauss-Seidel iteration |

### SIMPLE Algorithm Steps (per outer iteration)
1. Solve **u\*** from the u-momentum equation using current pressure `p`
2. Solve **v\*** from the v-momentum equation using current pressure `p`
3. Compute mass imbalance via **Rhie-Chow** face velocities
4. Solve the **pressure-correction** equation (Poisson, p')
5. Correct pressure: `p ← p + αp * p'`
6. Correct velocities: `u ← u* − (dy/aP) * ∂p'/∂x`
7. Repeat until convergence

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `nx`, `ny` | 61, 61 | Grid resolution |
| `nu` | 1e-2 | Kinematic viscosity → Re = 100 |
| `U_lid` | 1.0 m/s | Lid velocity |
| `urf_u`, `urf_v` | 0.5 | Momentum under-relaxation factors |
| `urf_p` | 0.2 | Pressure under-relaxation factor |
| `n_iter` | 500 | SIMPLE outer iterations |

To run at **Re = 400**, change `nu = 2.5e-3` (line commented in the code).

---

## Results & Validation

Results are validated against the benchmark data of **Ghia et al. (1982)** — the gold standard for lid-driven cavity flow at Re = 100. The solver plots:

- Pressure contour field with velocity quiver
- Streamlines overlaid on the pressure field
- Centreline **u-velocity** profile at x = 0.5 vs. Ghia data
- Centreline **v-velocity** profile at y = 0.5 vs. Ghia data

<img width="895" height="626" alt="image" src="https://github.com/user-attachments/assets/8f12c439-1b96-4ee5-9b1c-55020ecffd30" />
<img width="875" height="625" alt="image" src="https://github.com/user-attachments/assets/ffebb50e-0edc-4ffa-b1c0-1e466e0c41f3" />
<img width="567" height="453" alt="image" src="https://github.com/user-attachments/assets/6254e8e4-d0fa-455b-9c82-95d0d4c4c779" />
<img width="587" height="453" alt="image" src="https://github.com/user-attachments/assets/64bc79c8-d2b1-445f-95b4-fc6afd4715d8" />





---

## Requirements

```
numpy
matplotlib
```

Install via:
```bash
pip install numpy matplotlib
pip install numpy numpy
```

---

## Usage

```bash
python my_SIMPLE_with_comments.py
```

The solver will print iteration counts, max mass imbalance and display four plots upon completion.

---

## References

- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere Publishing.
- Ferziger, J.H. & Perić, M. (2002). *Computational Methods for Fluid Dynamics.* Springer.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
