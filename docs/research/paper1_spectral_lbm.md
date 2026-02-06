# Research Plan: Spectral-LBM Hybrid Methods for Turbulent Free-Surface Flows

## Paper 1: FFT-Based Spectral Acceleration

---

## 1. Proposed Title

**"Spectral Acceleration of Lattice Boltzmann Methods for Turbulent Multiphase Flows with Mass-Preserving Surface Smoothing"**

Alternative: *"Hybrid Spectral-LBM Methods: FFT-Based Subgrid Modeling and Interface Stabilization"*

---

## 2. Abstract (Draft)

We present a hybrid spectral-lattice Boltzmann method (LBM) that combines the locality and efficiency of LBM with the accuracy of Fourier spectral methods. Three key innovations are introduced: (1) a mass-preserving Helmholtz smoother for volume-of-fluid interfaces that eliminates spurious capillary currents while exactly conserving fluid mass; (2) a spectral strain-rate tensor computation for Smagorinsky-Lilly subgrid modeling that achieves spectral accuracy in eddy viscosity estimation; and (3) an exponential time differencing (ETD) scheme for thermal diffusion that removes CFL restrictions. The method is implemented on GPUs using VkFFT for OpenCL-based FFT operations. Validation against dam-break experiments and turbulent mixing benchmarks demonstrates improved stability at high Reynolds numbers and reduced spurious velocities at free surfaces compared to standard LBM approaches.

---

## 3. Motivation & Gap Analysis

### Current Limitations in LBM:
| Problem | Standard LBM Approach | Limitation |
|---------|----------------------|------------|
| Surface tension noise | PLIC reconstruction | O(Δx²) curvature errors → spurious currents |
| Subgrid turbulence | fneq-based Smagorinsky | Strain from non-equilibrium is approximate |
| Thermal diffusion | Explicit in collision | CFL-limited: Δt < Δx²/(2α) |
| High-k oscillations | None (local method) | Accumulate over time |

### Gap:
No existing work systematically applies spectral methods to LBM for:
- Interface smoothing with exact mass conservation
- Spectral-accurate strain tensor computation
- Unconditionally stable diffusion

---

## 4. Methodology

### 4.1 Spectral Surface Smoothing (SPECTRAL_SURFACE)

**Algorithm:**
```
Input: φ (fill level field), timestep t
if t % smooth_cadence ≠ 0: return

1. m_before = Σ φ[n]                    // Total mass before
2. φ_hat = FFT(φ)                       // Forward R2C FFT
3. φ_hat[k] *= 1/(1 + α|k|²)            // Helmholtz filter, H(0)=1
4. φ = IFFT(φ_hat)                      // Inverse FFT
5. m_after = Σ φ[n]                     // Total mass after
6. φ[n] -= (m_after - m_before)/N       // Exact mass correction
```

**Key properties:**
- H(0) = 1 preserves mean in exact arithmetic
- Explicit correction handles numerical drift
- Smoothing strength controlled by α and cadence

### 4.2 Spectral Subgrid Model (SPECTRAL_SUBGRID)

**Smagorinsky-Lilly model:**
```
ν_t = (C_s Δ)² |S|
```
where |S| = √(2 S_ij S_ij) is the strain rate magnitude.

**Spectral computation:**
```
1. û_hat = FFT(u_x), v̂_hat = FFT(u_y), ŵ_hat = FFT(u_z)
2. For each strain component S_ij:
   - Ŝ_ij[k] = 0.5(ik_j û_i + ik_i û_j)    // Spectral derivative
   - S_ij = IFFT(Ŝ_ij)                      // Back to physical space
   - Accumulate: |S|² += factor × S_ij²
3. ν_t = (C_s Δ)² √(2|S|²)
```

**Advantage over fneq-based:**
- Spectral accuracy for derivatives (vs O(Δx²) for finite difference equivalent)
- Decoupled from collision operator
- Configurable C_s Δ² parameter

### 4.3 ETD Thermal Diffusion (SPECTRAL_TEMPERATURE)

**Heat equation:** ∂T/∂t = α ∇²T

**Exact solution in Fourier space:**
```
T̂(k, t+Δt) = T̂(k, t) × exp(-α|k|²Δt)
```

**Algorithm:**
```
1. T_hat = FFT(T)
2. T_hat[k] *= exp(-α|k|²Δt)    // Exact integration
3. T = IFFT(T_hat)
```

**Properties:**
- Unconditionally stable (no CFL for diffusion)
- Exact for linear diffusion
- Preserves total thermal energy (∫T dx)

---

## 5. Implementation Details

### 5.1 Software Architecture
```
FluidX3D
├── src/
│   ├── spectral.hpp      // SpectralOps class declaration
│   ├── spectral.cpp      // VkFFT wrapper, kernels
│   ├── defines.hpp       // SPECTRAL_* toggles
│   ├── lbm.cpp           // Hook integration
│   └── kernel.cpp        // Modified SUBGRID block
└── docs/
    └── dev/spectral_accel_notes.md
```

### 5.2 Memory Layout
| Buffer | Size | Purpose |
|--------|------|---------|
| buffer_complex | (Nx/2+1)×Ny×Nz×2 | FFT output (interleaved Re/Im) |
| k_mag_sq | (Nx/2+1)×Ny×Nz | Pre-computed |k|² |
| ν_t | Nx×Ny×Nz | Eddy viscosity (SUBGRID only) |

### 5.3 GPU Implementation
- VkFFT for cross-platform FFT (OpenCL backend)
- Custom OpenCL kernels for spectral operations
- Parallel reduction for mass computation

---

## 6. Validation Experiments

### 6.1 Mass Conservation Test
**Setup:** Dam break in closed box
**Metric:** |m(t) - m(0)| / m(0) over 10⁵ timesteps
**Expected:** < 10⁻¹⁰ relative error with correction

### 6.2 Spurious Current Reduction
**Setup:** Static droplet under surface tension
**Metric:** max|u| (should be 0 for equilibrium)
**Compare:** Standard LBM vs. spectral smoothing
**Expected:** 10-100× reduction in spurious velocities

### 6.3 Taylor-Green Vortex Decay
**Setup:** 3D Taylor-Green at Re = 1600
**Metric:** Energy dissipation rate ε(t)
**Compare:**
- No subgrid model
- fneq-based Smagorinsky
- Spectral Smagorinsky
**Expected:** Spectral matches DNS reference better

### 6.4 Rayleigh-Bénard Convection
**Setup:** Heated bottom plate, cooled top
**Metric:** Nusselt number Nu(Ra)
**Compare:** Explicit vs. ETD diffusion
**Expected:** ETD allows 10× larger Δt for same accuracy

### 6.5 Dam Break with Turbulence
**Setup:** High-Re dam break (Re > 10⁵)
**Metrics:**
- Front position x_f(t)
- Surface smoothness (high-k energy)
**Expected:** Stable simulation without surface breakup artifacts

---

## 7. Expected Results & Contributions

### 7.1 Quantitative Claims
1. Mass conservation error < 10⁻¹⁰ (machine precision)
2. Spurious currents reduced by factor of 10-100
3. Spectral strain accuracy: O(N⁻ᵖ) vs O(Δx²) for fneq
4. ETD allows Δt up to 10× larger for thermal problems

### 7.2 Qualitative Contributions
1. First systematic FFT-LBM coupling for free surfaces
2. Novel mass-preserving spectral smoother
3. Spectral subgrid model for LBM turbulence
4. Open-source GPU implementation

---

## 8. Related Work

### LBM for Multiphase:
- Shan-Chen (1993): Pseudopotential model
- He-Chen-Zhang (1999): Free-energy LBM
- Latt et al. (2020): Palabos library

### Spectral Methods:
- Canuto et al. (2006): Spectral Methods textbook
- Boyd (2001): Chebyshev and Fourier Spectral Methods

### LBM Turbulence:
- Sagaut (2010): LES with LBM
- Malaspinas & Sagaut (2014): Wall-modeled LES-LBM

### Gap: No FFT-LBM hybrid for surfaces + turbulence

---

## 9. Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| 1 | 2 weeks | Literature review, refine methodology |
| 2 | 4 weeks | Run validation experiments |
| 3 | 2 weeks | Analyze results, generate figures |
| 4 | 3 weeks | Write manuscript |
| 5 | 2 weeks | Internal review, revisions |
| 6 | 1 week | Submit |

**Total: ~14 weeks**

---

## 10. Target Venues

### Primary:
1. **Journal of Computational Physics** (IF: 4.1)
   - Ideal for novel numerical methods
   - Strong CFD/multiphase community

### Secondary:
2. **Computer Physics Communications** (IF: 4.7)
   - Emphasis on implementation
   - Open-source code valued

3. **Physical Review E** (IF: 2.5)
   - Fluids section
   - Good for LBM community

### Conferences:
- DSFD (Discrete Simulation of Fluid Dynamics)
- ICMMES (Mesoscale Methods in Engineering and Science)

---

## 11. Authors & Contributions

| Author | Contribution |
|--------|--------------|
| [Lead] | Algorithm design, implementation, writing |
| [Co-author] | Validation experiments, analysis |
| [Advisor] | Supervision, manuscript review |

---

## 12. Code & Data Availability

- Code: GitHub (FluidX3D fork with spectral extensions)
- Data: Zenodo archive of validation results
- License: Open-source (match FluidX3D license)

---

## 13. Potential Reviewers

1. Jonas Latt (Palabos, Geneva)
2. Pierre Sagaut (LES-LBM, Aix-Marseille)
3. Timm Krüger (LBM multiphase, Edinburgh)
4. Orestis Malaspinas (LBM turbulence, Geneva)
5. Li-Shi Luo (LBM theory, ODU)

---

## 14. Risk Assessment

| Risk | Mitigation |
|------|------------|
| VkFFT compatibility issues | Fallback to clFFT or cuFFT |
| Multi-GPU not supported | Document as Phase 1 limitation |
| Spectral subgrid worse than fneq | Focus on surface smoothing contribution |
| Reviewer unfamiliar with spectral-LBM | Extensive background section |
