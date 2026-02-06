# Research Plan: Discontinuous Galerkin Methods for LBM Enhancement

## Paper 2: DG-LBM Hybrid Framework

---

## 1. Proposed Title

**"Discontinuous Galerkin Enhancement of Lattice Boltzmann Methods: Local High-Order Operators for Turbulent Multiphase Flows"**

Alternative: *"Beyond FFT: Discontinuous Galerkin Subgrid Modeling and Interface Reconstruction in LBM"*

---

## 2. Abstract (Draft)

We propose a novel discontinuous Galerkin (DG) enhanced lattice Boltzmann method that replaces global Fourier operations with local high-order polynomial operators. Unlike FFT-based approaches that require periodic boundaries, DG-LBM naturally handles complex geometries and wall-bounded flows while maintaining spectral-like accuracy. Three DG-enhanced operators are developed: (1) a conservative DG projection smoother for volume-of-fluid interfaces; (2) a DG-based strain tensor computation using modal derivatives; and (3) an implicit DG diffusion solver. The locality of DG enables efficient parallelization without distributed FFT communication, making the method suitable for exascale computing. Numerical experiments on turbulent channel flow, droplet collisions, and thermocapillary migration demonstrate that DG-LBM achieves comparable accuracy to spectral methods while handling non-periodic boundaries and adaptive mesh refinement.

---

## 3. Motivation & Gap Analysis

### Limitations of FFT-Based Spectral-LBM:
| Limitation | Impact | DG Solution |
|------------|--------|-------------|
| Requires periodicity | Can't do walls/inlets | Local basis functions |
| Global communication | MPI bottleneck at scale | Element-local operations |
| Uniform grid only | No AMR possible | hp-adaptivity natural |
| Gibbs phenomenon | Ringing near discontinuities | Slope limiters/WENO |

### Why DG for LBM?
1. **Shared philosophy:** Both DG and LBM are based on local operations with neighbor communication
2. **Conservation:** DG weak form conserves mass/momentum by construction
3. **High-order:** DG achieves spectral convergence for smooth solutions
4. **Flexibility:** Handles unstructured meshes, hanging nodes, p-refinement

---

## 4. Mathematical Framework

### 4.1 DG Basis Functions

**Modal basis (Legendre):**
```
ψ_p(ξ) = P_p(ξ),  ξ ∈ [-1, 1]
```
where P_p is the Legendre polynomial of degree p.

**3D tensor product:**
```
Ψ_{ijk}(ξ,η,ζ) = ψ_i(ξ) ψ_j(η) ψ_k(ζ)
```

**Field representation:**
```
φ(x) = Σ_{ijk} φ̂_{ijk} Ψ_{ijk}(ξ(x))
```

### 4.2 DG Derivative Operator

**Weak form:**
```
∫_K (∂φ/∂x) v dx = -∫_K φ (∂v/∂x) dx + ∫_{∂K} φ* v · n ds
```

**Matrix form:**
```
D_x φ̂ = M⁻¹ (S_x φ̂ + F_x(φ̂, φ̂_neighbors))
```
where:
- M = mass matrix
- S_x = stiffness matrix
- F_x = numerical flux (upwind/central)

### 4.3 DG Projection (Smoothing)

**L² projection onto polynomial space:**
```
φ_smooth = Π_p(φ) = argmin_{q ∈ P_p} ||φ - q||²_{L²(K)}
```

**Solution:**
```
φ̂_smooth = M⁻¹ ∫_K φ Ψ dx
```

**Properties:**
- Preserves moments up to degree p
- Removes modes > p (like low-pass filter)
- Exactly conserves mass (0th moment)

---

## 5. DG-LBM Hybrid Operators

### 5.1 DG Surface Smoother (vs FFT Helmholtz)

**Algorithm:**
```
Input: φ (fill level), polynomial degree p

For each element K:
  1. φ̂_K = project(φ_K, basis_p)     // Modal coefficients
  2. Apply modal filter:
     φ̂_K[mode] *= σ(mode/p)          // Exponential filter
  3. φ_K = evaluate(φ̂_K)             // Back to nodal

// Mass correction (element-wise or global)
4. Δm = Σ_K (m_K^new - m_K^old)
5. φ -= Δm / |Ω|
```

**Filter function:**
```
σ(η) = exp(-α η^s)  // Exponential filter, s=8-16 typical
```

**Advantages over FFT:**
- Handles walls (no periodicity needed)
- Local operation (parallelizes trivially)
- Adaptive: higher p near interface

### 5.2 DG Strain Tensor (vs FFT Spectral Derivatives)

**Velocity gradient via DG:**
```
∂u_i/∂x_j = D_j u_i = M⁻¹ (S_j û_i + Flux_j)
```

**Strain rate:**
```
S_ij = 0.5 (∂u_i/∂x_j + ∂u_j/∂x_i)
```

**Strain magnitude:**
```
|S| = √(2 S_ij S_ij)
```

**DG implementation:**
```
For each element K:
  1. Compute ∂u_i/∂x_j for all i,j using DG derivative
  2. S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i)
  3. |S|² = 2(S_xx² + S_yy² + S_zz² + 2S_xy² + 2S_xz² + 2S_yz²)
  4. ν_t = (C_s Δ)² |S|
```

**Flux choice:**
- Central: 6th order for smooth flows
- Upwind: More stable, slight dissipation
- BR1/BR2: Bassi-Rebay for viscous terms

### 5.3 DG Implicit Diffusion (vs FFT ETD)

**Heat equation:** ∂T/∂t = α ∇²T

**DG weak form:**
```
∫_K (∂T/∂t) v dx = -α ∫_K ∇T · ∇v dx + α ∫_{∂K} (∇T·n)* v ds
```

**Implicit time stepping:**
```
(M + αΔt K) T̂^{n+1} = M T̂^n + BC terms
```
where K is the DG Laplacian matrix.

**Block structure:**
- Each element K has local (p+1)³ × (p+1)³ system
- Coupling only through face fluxes
- Solve with block Jacobi + GMRES

**Advantages:**
- Unconditionally stable (implicit)
- Handles Dirichlet/Neumann BC naturally
- No periodicity requirement

---

## 6. Theoretical Analysis

### 6.1 Convergence Rates

| Quantity | DG Order p | Convergence |
|----------|------------|-------------|
| φ (smooth) | p | O(h^{p+1}) |
| ∂φ/∂x | p | O(h^p) |
| Projection error | p | O(h^{p+1}) |

For p → ∞, exponential convergence (spectral).

### 6.2 Conservation Properties

**Mass (φ):**
```
d/dt ∫_Ω φ dx = ∫_{∂Ω} flux ds = 0 (closed domain)
```
DG preserves exactly via consistent fluxes.

**Momentum (ρu):**
LBM collision preserves; DG derivatives are conservative.

### 6.3 Stability Analysis

**CFL condition for DG:**
```
Δt ≤ C h / ((2p+1) |u|_max)
```
where C ~ 1 depends on Runge-Kutta scheme.

**For implicit diffusion:** Unconditionally stable.

---

## 7. Implementation Strategy

### 7.1 Software Architecture
```
FluidX3D-DG/
├── src/
│   ├── dg_basis.hpp       // Legendre polynomials, quadrature
│   ├── dg_operators.hpp   // Mass, stiffness, flux matrices
│   ├── dg_element.hpp     // Element class with local solve
│   ├── dg_mesh.hpp        // Connectivity, halos
│   ├── dg_lbm.cpp         // DG-LBM coupling
│   └── dg_kernels.cl      // OpenCL kernels
└── tests/
    ├── test_dg_derivative.cpp
    ├── test_dg_projection.cpp
    └── test_dg_diffusion.cpp
```

### 7.2 Data Structures

**Per-element storage:**
```cpp
struct DG_Element {
    float coeffs[(P+1)³];     // Modal coefficients
    float nodes[(P+1)³];      // Nodal values
    uint neighbors[6];        // Face neighbor indices
    float face_flux[6][(P+1)²]; // Pre-computed fluxes
};
```

### 7.3 GPU Kernels

**Projection kernel:**
```opencl
kernel void dg_project(
    global float* phi_nodal,    // Input: LBM field
    global float* phi_modal,    // Output: DG coefficients
    constant float* M_inv,      // Inverse mass matrix
    constant float* basis_vals  // Basis at quadrature points
) {
    uint elem = get_group_id(0);
    uint mode = get_local_id(0);

    // Integrate: φ̂[mode] = ∫ φ Ψ[mode] dx
    float integral = 0.0f;
    for (uint q = 0; q < N_QUAD; q++) {
        integral += phi_nodal[elem*N_QUAD + q]
                  * basis_vals[mode*N_QUAD + q]
                  * quad_weights[q];
    }

    // Apply M⁻¹
    phi_modal[elem*N_MODES + mode] = M_inv[mode] * integral;
}
```

---

## 8. Validation Experiments

### 8.1 Convergence Study
**Setup:** Smooth manufactured solution
**Vary:** Polynomial degree p = 1, 2, 4, 8
**Metric:** L² error vs. degrees of freedom
**Expected:** Exponential convergence for high p

### 8.2 Channel Flow (Non-Periodic Test)
**Setup:** Poiseuille flow with wall BC
**Compare:**
- Standard LBM
- FFT-LBM (can't do walls!)
- DG-LBM with p=4
**Metric:** Velocity profile accuracy

### 8.3 Turbulent Channel (Re_τ = 180)
**Setup:** DNS-like resolution
**Compare:** DG subgrid vs fneq Smagorinsky
**Metrics:**
- Mean velocity u⁺(y⁺)
- Reynolds stress <u'v'>
**Validation:** Moser et al. (1999) DNS data

### 8.4 Droplet on Wall
**Setup:** Sessile droplet with contact angle
**Compare:** DG smoothing vs FFT (FFT fails at wall)
**Metric:** Contact angle accuracy, spurious currents

### 8.5 Adaptive Refinement
**Setup:** Rising bubble with p-refinement near interface
**Demonstrate:** DG enables local high-order where needed
**Metric:** Accuracy vs. DOF count

---

## 9. Comparison: FFT vs DG

### 9.1 Theoretical Comparison

| Aspect | FFT-LBM | DG-LBM |
|--------|---------|--------|
| Accuracy | Spectral | Spectral (high p) |
| Boundaries | Periodic only | Any BC |
| Parallelism | Global comm (AllToAll) | Local (neighbor only) |
| Memory | O(N log N) work buffers | O(p³) per element |
| AMR | Not possible | Natural |
| Implementation | Simple (library call) | Complex (custom code) |

### 9.2 Computational Cost

**FFT:**
- Forward: O(N log N)
- Filter: O(N)
- Inverse: O(N log N)
- Total: O(N log N)

**DG (per element):**
- Project: O(p³ × Q) where Q = quadrature points
- Derivative: O(p⁴) for matrix-vector
- Total: O(N_elem × p⁴)

**Crossover:** DG cheaper when N_elem × p⁴ < N log N
- For N = 256³, p = 4: DG competitive
- For N = 512³, p = 2: FFT faster

### 9.3 Hybrid Strategy

Use **FFT for interior**, **DG near boundaries**:
```
if (element.touches_boundary)
    apply_dg_smoother(element)
else
    apply_fft_smoother(element)  // Needs careful stitching
```

---

## 10. Novel Contributions

1. **First DG-LBM hybrid for multiphase flows**
   - DG projection as interface smoother
   - Conservative by construction

2. **DG-based Smagorinsky for LBM**
   - High-order strain tensor
   - Wall-compatible (no periodicity)

3. **Implicit DG diffusion in LBM**
   - Block-structured solver
   - Handles thermal BC

4. **Comparison framework: FFT vs DG**
   - When to use which
   - Hybrid strategies

---

## 11. Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| 1 | 3 weeks | DG infrastructure (basis, matrices) |
| 2 | 4 weeks | DG operators (project, derivative, diffusion) |
| 3 | 3 weeks | LBM integration and testing |
| 4 | 4 weeks | Validation experiments |
| 5 | 2 weeks | FFT vs DG comparison |
| 6 | 3 weeks | Write manuscript |
| 7 | 2 weeks | Revisions |

**Total: ~21 weeks**

---

## 12. Target Venues

### Primary:
1. **SIAM Journal on Scientific Computing** (IF: 3.1)
   - Strong DG community
   - Appreciates numerical analysis

### Secondary:
2. **Journal of Computational Physics** (IF: 4.1)
   - Multiphysics focus
   - Good for hybrid methods

3. **Computer Methods in Applied Mechanics and Engineering** (IF: 7.2)
   - Engineering applications
   - High impact

### Conferences:
- SIAM CSE (Computational Science & Engineering)
- ICOSAHOM (High-Order Methods)
- WCCM (World Congress on Computational Mechanics)

---

## 13. Technical Challenges & Mitigations

| Challenge | Risk | Mitigation |
|-----------|------|------------|
| DG Laplacian stability | Interior penalty tuning | Use proven BR2 scheme |
| LBM-DG grid mismatch | Interpolation errors | Co-located nodes |
| GPU efficiency for DG | Irregular memory access | SoA layout, coalescing |
| High-p cost | O(p⁴) expensive | Limit to p ≤ 8 |
| Complex geometry meshing | Time-consuming | Start with Cartesian, add later |

---

## 14. Code Development Plan

### Phase 1: DG Core (Weeks 1-3)
- [ ] Legendre polynomial evaluation
- [ ] Gauss-Lobatto quadrature
- [ ] Mass/stiffness matrix assembly
- [ ] Unit tests for basis functions

### Phase 2: DG Operators (Weeks 4-7)
- [ ] L² projection
- [ ] Modal filtering
- [ ] DG derivative (central flux)
- [ ] DG Laplacian (BR2)
- [ ] Unit tests for operators

### Phase 3: LBM Integration (Weeks 8-10)
- [ ] DG smoother for φ
- [ ] DG strain tensor → ν_t
- [ ] DG diffusion for T
- [ ] Hook into FluidX3D

### Phase 4: Validation (Weeks 11-14)
- [ ] Convergence study
- [ ] Channel flow
- [ ] Droplet on wall
- [ ] Turbulent channel

### Phase 5: Comparison (Weeks 15-16)
- [ ] FFT vs DG benchmarks
- [ ] Scaling study
- [ ] Hybrid strategy exploration

---

## 15. Expected Outcomes

### Publications:
1. Main paper: DG-LBM methodology (this plan)
2. Follow-up: Adaptive DG-LBM for bubble dynamics
3. Software paper: Open-source DG-LBM library

### Software:
- FluidX3D-DG extension (open source)
- Reusable DG operator library

### Broader Impact:
- Enables LBM for complex geometries
- Path to exascale (local operations)
- Foundation for hp-adaptive LBM

---

## 16. Future Work (Beyond This Paper)

1. **ADER-DG-LBM:** High-order time integration
2. **hp-Adaptivity:** Automatic p-refinement near interfaces
3. **Curved elements:** For complex geometry
4. **Multi-GPU DG:** Distributed memory parallelism
5. **Machine learning:** Neural network for optimal filter

---

## 17. References (Key)

### DG Methods:
- Hesthaven & Warburton (2008): Nodal DG Methods
- Cockburn et al. (2000): DG for CFD
- Kopriva (2009): Implementing Spectral Methods

### DG for Incompressible Flow:
- Shahbazi et al. (2007): DG for Navier-Stokes
- Ferrer & Willden (2012): DG-LES

### LBM + High-Order:
- Ubertini et al. (2004): Finite-volume LBM
- Fakhari & Lee (2014): High-order LBM

### Gap: No DG + LBM for multiphase/turbulence
