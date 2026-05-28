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

// ---- FluidX3D <-> Lallemand-Luo D2Q9 direction translation ----
//
// FluidX3D D2Q9 ordering (from kernel.cpp lines 867-869, antipode-paired):
//   0=(0,0), 1=(+x,0), 2=(-x,0), 3=(0,+y), 4=(0,-y),
//   5=(+x,+y), 6=(-x,-y), 7=(+x,-y), 8=(-x,+y)
//
// Lallemand-Luo ordering (this file, basis-orthogonal):
//   0=(0,0), 1=(+x,0), 2=(0,+y), 3=(-x,0), 4=(0,-y),
//   5=(+x,+y), 6=(-x,+y), 7=(-x,-y), 8=(+x,-y)
//
// fx_to_ll[i] = LL index for FluidX3D direction i:
//   {0, 1, 3, 2, 4, 5, 7, 8, 6}
inline constexpr int fx_to_ll[Q] = {0, 1, 3, 2, 4, 5, 7, 8, 6};
inline constexpr int ll_to_fx[Q] = {0, 1, 3, 2, 4, 5, 8, 6, 7};

// Translate a 9-vector indexed by FluidX3D ordering into Lallemand-Luo
// ordering. `f_LL[k]` receives the value of `f_FX[ll_to_fx[k]]`.
void fluidx3d_to_LL_D2Q9(const float f_FX[Q], double f_LL[Q]);

// ---- Esoteric-Pull post-collision DDF reconstruction ----
//
// FluidX3D stores DDFs with the Esoteric-Pull in-place streaming scheme, so
// the raw device buffer is NOT a naive fi[i*N + n] SoA array of post-collision
// populations. store_f() (kernel.cpp) scatters each post-collision population
// fhn[i] to a location that depends on the antipode pairing, the neighbor
// index j[i], and the time-step parity. To recover cell n's post-collision
// populations on the host we must invert store_f() exactly.
//
// This reconstructs the post-collision DDF vector fhn[] (FluidX3D direction
// ordering) for the 2D cell (x, y), reading from the raw device buffer `fi`
// (FP32, index_f(n,i) = i*N + n). `t_store` is the time-step value that was
// passed to store_f() for the data currently in `fi` -- i.e. get_t() - 1 after
// do_time_step() returns. Periodic boundaries in x and y, matching neighbors().
void reconstruct_post_collision_D2Q9(const float* fi, unsigned long N,
                                     int x, int y, int Nx, int Ny,
                                     unsigned long long t_store,
                                     float fhn_FX[Q]);

// Inverse: pack a Lallemand-Luo-ordered 9-vector into FluidX3D ordering.
// Used in synthetic-test construction only.
void LL_to_fluidx3d_D2Q9(const double f_LL[Q], float f_FX[Q]);

// ---- V1 hook (host-side, called from LBM::run() per step) ----

// Diagnostic state carried between time steps for the V1 gate.
// Configure once in main_setup() (csv_path, geometry, u_w, tau_plus);
// the LBM time loop drives v1_hook_tick() once per step.
struct V1Hook {
    // Configuration (must be set before first sample)
    const char* csv_path = nullptr;   // if null, hook is disabled
    int Nx = 0, Ny = 0;               // 2D grid dimensions
    int corner_buffer = 4;            // exclude this many cells from each x-edge of the wall
    int wall_y = 0;                   // y-coordinate of the wall-adjacent fluid cell (e.g., Ny-2 for top wall)
    double u_w = 0.0;                 // top-wall velocity in lattice units
    double tau_plus = 0.0;            // BGK relaxation time
    unsigned long long sample_step_start = 100000ULL;
    unsigned long long sample_cadence  = 5000ULL;
    const char* build_hash = "unknown";

    // Internal state
    void* csv_handle = nullptr;       // FILE* (kept as void* to keep <cstdio> out of header)
    unsigned int n_samples_taken = 0;
    bool readback_validated = false;  // set true after the one-time rho/u cross-check
};

// Cheap test: would `step` trigger a sample? Lets the caller avoid an expensive
// full-buffer device read-back on non-sample steps.
bool v1_hook_should_sample(const V1Hook& hook, unsigned long long step);

// Sample the wall-adjacent row of cells at `step` (caller should gate on
// v1_hook_should_sample first). Returns true if a sample was taken and
// appended to CSV.
//
// `fi_raw` is the raw Esoteric-Pull device buffer (FP32), read back to host;
// reconstruct_post_collision_D2Q9() recovers the true post-collision
// populations from it. `t_store` = step - 1 (the t passed to store_f).
//
// If `rho_fx` and `u_fx` are non-null, the FIRST sample additionally validates
// the read-back: it reconstructs rho/u at a row of interior cells and checks
// them against FluidX3D's own device fields (which use the correct indexing).
// `rho_fx[n]` is density; `u_fx` is AoS with u_x at [0,N), u_y at [N,2N).
bool v1_hook_tick(V1Hook& hook,
                  unsigned long long step,
                  const float* fi_raw,
                  unsigned long N,
                  const float* rho_fx = nullptr,
                  const float* u_fx = nullptr);

// Close the CSV. Called manually at end of simulation; safe to skip on crash.
void v1_hook_close(V1Hook& hook);

// Shared V1 hook instance. setup.cpp configures fields before lbm.run();
// LBM::run() drives v1_hook_tick() once per step against this instance.
// When csv_path == nullptr (default), the hook is disabled.
extern V1Hook g_v1_hook;

} // namespace paper3
