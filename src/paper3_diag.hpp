// paper3_diag.hpp -- Paper 3 V1 ghost-energy hook math kernel.
//
// D2Q9 d'Humieres moment transform + Paper 2 Phase A Lock 3 normalized
// ghost-energy projection onto the full non-conserved subspace. Drives
// the V1 gate diagnostic eps_g_hat = 5/18 on a Ladd-injected face wall.
//
// Math conventions locked in:
//   Pavlov-Number-Windows/plans/paper3_d2q9_face_wall_derivation.md
//
// Verified by src/paper3_diag_selftest.cpp (20/20 passing, target
// eps_g_hat = 5/18 to machine precision).

#pragma once

namespace paper3 {

constexpr int Q = 9;
constexpr double cs2 = 1.0 / 3.0;

// Direction ordering (Lallemand-Luo 2000):
//   0=(0,0), 1=(+x), 2=(+y), 3=(-x), 4=(-y),
//   5=(+x,+y), 6=(-x,+y), 7=(-x,-y), 8=(+x,-y)
inline constexpr double c_x[Q] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
inline constexpr double c_y[Q] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
inline constexpr double w_i[Q] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                                  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

// d'Humieres moment basis (Lallemand-Luo 2000, Table I).
// Row k = mode; column i = direction.
// Modes: 0=rho, 1=e, 2=eps, 3=jx, 4=qx, 5=jy, 6=qy, 7=pxx, 8=pxy
inline constexpr int h[Q][Q] = {
    //   i=0,  1,  2,  3,  4,  5,  6,  7,  8
    {     1,  1,  1,  1,  1,  1,  1,  1,  1},  // m_0 = rho
    {    -4, -1, -1, -1, -1,  2,  2,  2,  2},  // m_1 = e
    {     4, -2, -2, -2, -2,  1,  1,  1,  1},  // m_2 = epsilon
    {     0,  1,  0, -1,  0,  1, -1, -1,  1},  // m_3 = j_x
    {     0, -2,  0,  2,  0,  1, -1, -1,  1},  // m_4 = q_x
    {     0,  0,  1,  0, -1,  1,  1, -1, -1},  // m_5 = j_y
    {     0,  0, -2,  0,  2,  1,  1, -1, -1},  // m_6 = q_y
    {     0,  1, -1,  1, -1,  0,  0,  0,  0},  // m_7 = p_xx
    {     0,  0,  0,  0,  0,  1, -1,  1, -1},  // m_8 = p_xy
};

// Conserved-mode indices: rho (0), j_x (3), j_y (5). Excluded from ghost projection.
inline constexpr int CONSERVED[3] = {0, 3, 5};
// Full non-conserved subspace (the V1 gate projector).
inline constexpr int NON_CONSERVED[6] = {1, 2, 4, 6, 7, 8};
// Narrow-sense Lallemand-Luo "ghost" subset (subspace-bug control only;
// must NOT be used for the V1 gate -- see checklist §6 failure row).
inline constexpr int NARROW_GHOST[4] = {1, 2, 4, 6};

constexpr bool is_conserved(int k) { return k == 0 || k == 3 || k == 5; }

// ---- math kernel ----

// Weighted moment norm ||h_k||_w^2 = sum_i w_i h[k][i]^2.
double norm_w_sq(int k);

// Unweighted moment norm ||h_k||^2 = sum_i h[k][i]^2 (Lallemand-Luo orthogonality).
double norm_sq(int k);

// Standard D2Q9 equilibrium, second order in u.
void compute_feq(double rho, double ux, double uy, double feq[Q]);

// Forward moment transform: m_k = sum_i h[k][i] f_i.
void moment_transform(const double f[Q], double m[Q]);

// Full V1-gate pipeline:
//   1. rho, j_x, j_y from f.
//   2. f_neq = f - f^eq(rho, u).
//   3. m_k = moments of f_neq.
//   4. eps_g_bnd = (1/(2 c_s^2)) * (1/2) * sum_{k in subspace} m_k^2 / ||h_k||_w^2.
//   5. eps_g_hat = eps_g_bnd / (tau^2 Ma^2).
//
// The 1/(2 c_s^2) prefactor matches Paper 2 Phase A Lock 3 -- without it
// the V1 gate lands at 5/27 instead of 5/18 (off by 3/2). See
// plans/paper3_d2q9_face_wall_derivation.md.
//
// `subspace` enumerates which moment indices contribute. For the V1 gate
// pass NON_CONSERVED with n_subspace=6.
double compute_eps_g_hat(const double f[Q], double tau_plus, double Ma,
                         const int* subspace, int n_subspace);

// Convenience wrapper using the full non-conserved subspace (V1 gate definition).
double compute_eps_g_hat_full(const double f[Q], double tau_plus, double Ma);

// ---- off-line verification helper ----

// Build synthetic post-collision populations representing the steady-state
// Ladd-injected ghost at a top wall with u_w = (u_w, 0):
//   m_{q_x}^ss  = +tau_plus * u_w / 3
//   m_{p_xy}^ss = -tau_plus * u_w / 3
// (Other non-conserved modes have zero injection for this geometry.)
// Base state: rho = 1, u = 0 -- isolates the kernel from macroscopic flow.
//
// Used by paper3_diag_selftest.cpp; NOT part of the real V1 LBM run.
void build_synthetic_ladd_top_wall_x(double u_w, double tau_plus, double f_synth[Q]);

} // namespace paper3
