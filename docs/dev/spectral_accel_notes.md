# Spectral Accelerators for FluidX3D Mixing Simulations

## Overview

This document describes the FFT-based spectral acceleration infrastructure for FluidX3D,
targeting free-surface mixing simulations with strong subgrid turbulence.

Based on techniques from:
- moljax: GPU-Accelerated Stiff Transport-Reaction PDE Simulation (Pavlov)
- CFL-informed NILT Parameter Selection (Pavlov)

---

## 1. Arrays to Smooth (SPECTRAL_SURFACE)

### Target: `phi` (fill level)

| Array | Type | Size | Location | Purpose |
|-------|------|------|----------|---------|
| `phi` | `Memory<float>` | N | `lbm.hpp:63` | Volume-of-fluid fill level [0,1] |

**Why smooth phi?**
- PLIC curvature reconstruction amplifies high-k noise in phi
- Surface tension forces depend on curvature -> noisy forces -> instability
- Spectral smoothing dampens spurious high-frequency oscillations

**Smoothing approach:**
- Helmholtz filter: `phi_hat *= 1/(1 + alpha*|k|^2)`
- H(0) = 1 guarantees mean preservation
- Post-FFT mass correction: `phi -= mean(phi_new) - mean(phi_old)`

**Hook point:** After `enqueue_surface_3()` in `lbm.cpp:904`

```
surface_3 updates phi -> spectral smooth phi -> communicate_phi_massex_flags
```

---

## 2. Scalars to Compute (SPECTRAL_SUBGRID)

### Target: `nu_t` (eddy viscosity)

| Array | Type | Size | Location | Purpose |
|-------|------|------|----------|---------|
| `nu_t` | `Memory<float>` | N | NEW | Turbulent eddy viscosity |

**Why NOT store full gradient tensor?**
- Full tensor = 9 floats/cell = +36 bytes/cell
- FluidX3D is memory-bandwidth optimized (even uses FP16 DDFs)
- Storing only nu_t = +4 bytes/cell (9x smaller!)

**Computation approach:**
1. FFT velocity components (ux, uy, uz)
2. Compute strain rate tensor in k-space: `S_ij_hat = 0.5*(i*k_j*u_i_hat + i*k_i*u_j_hat)`
3. Compute strain magnitude: `|S|_hat` from contracted tensor
4. IFFT to get |S| in physical space
5. Compute: `nu_t = (C_s * delta)^2 * |S|` where C_s ~ 0.1-0.2

**Hook point:** Before `enqueue_stream_collide()` in `lbm.cpp:895`

```
compute nu_t spectrally -> stream_collide uses nu_t for modified relaxation
```

**Kernel modification:** In `kernel.cpp:1600-1620`, spectral nu_t is used when SPECTRAL_SUBGRID enabled:
```cpp
#ifdef SPECTRAL_SUBGRID
{ // Use spectrally-computed eddy viscosity nu_t
    const float tau0 = 1.0f/w;
    const float nu_turb = nu_t[n]; // read spectral eddy viscosity
    w = 1.0f/(tau0 + 3.0f*nu_turb); // tau_eff = tau0 + 3*nu_t
}
#else
// Original inline Smagorinsky from fneq
#endif
```

**Note:** The relationship is `tau = 3*nu + 0.5`, so `tau_eff = tau0 + 3*nu_t` for `nu_eff = nu_0 + nu_t`.

---

## 3. Hook Points in do_time_step()

Location: `src/lbm.cpp` lines 891-920

```cpp
void LBM::do_time_step() {
    // --- SURFACE phase 0 ---
    for(d) enqueue_surface_0();           // Capture outgoing DDFs

    // *** SPECTRAL_SUBGRID hook (before collision) ***
    // Compute spectral eddy viscosity here
    // nu_t must be ready before stream_collide uses it

    // --- Main LBM kernel ---
    for(d) enqueue_stream_collide();      // Collision + streaming (uses nu_t if SPECTRAL_SUBGRID)

    // --- Communications ---
    communicate_rho_u_flags();

    // --- SURFACE phases 1-3 ---
    for(d) enqueue_surface_1();           // Prevent gas neighbors
    communicate_flags();
    for(d) enqueue_surface_2();           // Apply flag changes
    communicate_flags();
    for(d) enqueue_surface_3();           // Mass/phi update

    // *** SPECTRAL_SURFACE hook (after surface_3) ***
    // Smooth phi here, before communication
    // Must preserve mass exactly

    communicate_phi_massex_flags();

    // --- TEMPERATURE ---
    communicate_fi();
    communicate_gi();

    // *** SPECTRAL_TEMPERATURE hook (after gi comm) ***
    // Apply IMEX/ETD diffusion step here

    // --- Finalize ---
    for(d) enqueue_integrate_particles();
    finish_queue();
    increment_time_step();
}
```

---

## 4. Periodicity Requirements

### Fully Periodic (all dims)
- Standard 3D R2C FFT via VkFFT
- Most efficient, recommended for periodic box simulations

### Partial Periodic (e.g., periodic x/y, walls at z)
- Use 2D FFT per z-slice
- Finite differences for z-derivatives in SUBGRID
- Common for channel-type mixing with top/bottom walls

### Non-Periodic (walls everywhere)
- Spectral ops disabled by default
- Future: DCT-based smoothing (VkFFT supports R2R transforms)

**Runtime detection:**
```cpp
if (!is_periodic_x() || !is_periodic_y() || !is_periodic_z()) {
    spectral->configure_partial_spectral(periodic_x, periodic_y, periodic_z);
}
```

---

## 5. Memory Layout

### Existing FluidX3D layout
- Linear indexing: `n = x + (y + z*Ny)*Nx`
- Velocity AoS: `u[n]`, `u[N+n]`, `u[2*N+n]` for components
- phi: `phi[n]` single float per cell

### FFT buffer layout (R2C)
- Input: `Nx * Ny * Nz` real floats
- Output: `(Nx/2+1) * Ny * Nz` complex values
- Storage: `(Nx/2+1) * Ny * Nz * 2` floats (interleaved Re/Im)

### Wavenumber arrays
- `kx[i]` for i in [0, Nx/2]: `i * 2*pi/Nx`
- `ky[j]` for j in [0, Ny): standard FFT ordering (0..Ny/2, -Ny/2+1..-1)
- `kz[k]` for k in [0, Nz): standard FFT ordering
- `k_mag_sq[(Nx/2+1)*Ny*Nz]`: pre-computed |k|^2

---

## 6. Compile-Time Toggles

Add to `src/defines.hpp` after line 25:

```cpp
//#define SPECTRAL_SURFACE     // FFT-based phi smoothing
//#define SPECTRAL_SUBGRID     // Spectral eddy viscosity
//#define SPECTRAL_TEMPERATURE // IMEX/ETD diffusion (future)

// Dependency checks
#ifdef SPECTRAL_SURFACE
#ifndef SURFACE
#error "SPECTRAL_SURFACE requires SURFACE"
#endif
#endif

#ifdef SPECTRAL_SUBGRID
#ifndef SUBGRID
#error "SPECTRAL_SUBGRID requires SUBGRID"
#endif
#define UPDATE_FIELDS  // Force velocity update every step
#endif

// Master toggle
#if defined(SPECTRAL_SURFACE) || defined(SPECTRAL_SUBGRID) || defined(SPECTRAL_TEMPERATURE)
#define SPECTRAL_OPS
#endif

// Tuning parameters
#ifndef SPECTRAL_SMOOTH_EVERY
#define SPECTRAL_SMOOTH_EVERY 8u
#endif

#ifndef SPECTRAL_HELMHOLTZ_ALPHA
#define SPECTRAL_HELMHOLTZ_ALPHA 1.0f  // Smoothing length scale
#endif
```

---

## 7. Memory Budget

| Component | bytes/cell | Condition |
|-----------|------------|-----------|
| Complex buffer | ~4.1 | SPECTRAL_OPS (shared) |
| k_mag_sq | ~2.1 | SPECTRAL_OPS (shared) |
| nu_t | 4.0 | SPECTRAL_SUBGRID only |

**Totals:**
- SPECTRAL_SURFACE only: ~6 bytes/cell
- SPECTRAL_SURFACE + SPECTRAL_SUBGRID: ~10 bytes/cell

Compare to original SUBGRID: 0 extra bytes (computes on-the-fly from fneq)

---

## 8. Validation Diagnostics

### Mass Conservation (SURFACE)
```cpp
float mass_before = reduce_sum(phi);
smooth_phi(...);
float mass_after = reduce_sum(phi);
assert(fabs(mass_after - mass_before) < 1e-5f * N);
```

### High-k Energy (SURFACE)
```cpp
float E_high = compute_high_k_energy(phi, k_cutoff);
// Track over time: should decrease with smoothing
```

### Eddy Viscosity Comparison (SUBGRID)
```cpp
// Compare spectral nu_t with original Smagorinsky
// Should match within ~10% for resolved scales
```

---

## 9. Files to Create

| File | Purpose |
|------|---------|
| `src/spectral.hpp` | SpectralOps class declaration |
| `src/spectral.cpp` | VkFFT wrapper implementation |
| `src/spectral_kernels.cpp` | OpenCL kernels for k-space ops |

## 10. Files to Modify

| File | Changes |
|------|---------|
| `src/defines.hpp` | Add SPECTRAL_* toggles |
| `src/lbm.hpp` | Add SpectralOps member, nu_t buffer |
| `src/lbm.cpp` | Add allocation, hook calls |
| `src/kernel.cpp` | Modify SUBGRID block for spectral nu_t |
| `makefile` | Add spectral.o target, VkFFT includes |
