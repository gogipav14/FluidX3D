// paper3_diag.cpp -- Implementation of paper3_diag.hpp.
//
// Math conventions: plans/paper3_d2q9_face_wall_derivation.md
// V1 usage:         plans/paper3_phase_B_v1_checklist.md

#include "paper3_diag.hpp"

#include <cmath>
#include <cstdio>

namespace paper3 {

double norm_w_sq(int k) {
    double s = 0.0;
    for (int i = 0; i < Q; ++i) s += w_i[i] * h[k][i] * h[k][i];
    return s;
}

double norm_sq(int k) {
    double s = 0.0;
    for (int i = 0; i < Q; ++i) s += (double)h[k][i] * h[k][i];
    return s;
}

void compute_feq(double rho, double ux, double uy, double feq[Q]) {
    const double u2 = ux * ux + uy * uy;
    for (int i = 0; i < Q; ++i) {
        const double cu = c_x[i] * ux + c_y[i] * uy;
        feq[i] = w_i[i] * rho * (1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * u2 / cs2);
    }
}

void moment_transform(const double f[Q], double m[Q]) {
    for (int k = 0; k < Q; ++k) {
        double s = 0.0;
        for (int i = 0; i < Q; ++i) s += h[k][i] * f[i];
        m[k] = s;
    }
}

double compute_eps_g_hat(const double f[Q], double tau_plus, double Ma,
                         const int* subspace, int n_subspace,
                         bool subtract_rest) {
    double rho = 0.0, jx = 0.0, jy = 0.0;
    for (int i = 0; i < Q; ++i) {
        rho += f[i];
        jx  += c_x[i] * f[i];
        jy  += c_y[i] * f[i];
    }
    const double ux = jx / rho;
    const double uy = jy / rho;

    double feq[Q];
    // Rest reference (Paper 1 kick): subtract f_eq(rho, 0) so f_neq is the
    // injection relative to rest. Local reference (Paper 2 F_gh): subtract
    // f_eq(rho, u_local). See header.
    if (subtract_rest) compute_feq(rho, 0.0, 0.0, feq);
    else               compute_feq(rho, ux, uy, feq);

    double f_neq[Q];
    for (int i = 0; i < Q; ++i) f_neq[i] = f[i] - feq[i];

    double m[Q];
    moment_transform(f_neq, m);

    double sum = 0.0;
    for (int s = 0; s < n_subspace; ++s) {
        const int k = subspace[s];
        sum += (m[k] * m[k]) / norm_w_sq(k);
    }

    const double eps_g_bnd = (1.0 / (2.0 * cs2)) * 0.5 * sum;
    return eps_g_bnd / (tau_plus * tau_plus * Ma * Ma);
}

double compute_eps_g_hat_full(const double f[Q], double tau_plus, double Ma,
                              bool subtract_rest) {
    return compute_eps_g_hat(f, tau_plus, Ma, NON_CONSERVED, 6, subtract_rest);
}

void build_synthetic_ladd_top_wall_x(double u_w, double tau_plus, double f_synth[Q]) {
    double feq[Q];
    compute_feq(1.0, 0.0, 0.0, feq);

    const double m_qx_ss  = +tau_plus * u_w / 3.0;
    const double m_pxy_ss = -tau_plus * u_w / 3.0;
    const double N_qx  = norm_sq(4);   // = 12 (unweighted Lallemand-Luo)
    const double N_pxy = norm_sq(8);   // = 4

    for (int i = 0; i < Q; ++i) {
        f_synth[i] = feq[i]
                   + h[4][i] * m_qx_ss  / N_qx
                   + h[8][i] * m_pxy_ss / N_pxy;
    }
}

// ---- FluidX3D <-> Lallemand-Luo translation ----

void fluidx3d_to_LL_D2Q9(const float f_FX[Q], double f_LL[Q]) {
    for (int i = 0; i < Q; ++i) f_LL[fx_to_ll[i]] = (double) f_FX[i];
}

void LL_to_fluidx3d_D2Q9(const double f_LL[Q], float f_FX[Q]) {
    for (int k = 0; k < Q; ++k) f_FX[ll_to_fx[k]] = (float) f_LL[k];
}

// ---- Esoteric-Pull post-collision reconstruction ----

void reconstruct_post_collision_D2Q9(const float* fi, unsigned long N,
                                     int x, int y, int Nx, int Ny,
                                     unsigned long long t_store,
                                     float fhn_FX[Q]) {
    // FluidX3D D2Q9 neighbor indices with periodic BC (matches neighbors() in
    // kernel.cpp). n = x + y*Nx for the 2D (Nz=1) case.
    const unsigned long x0 = (unsigned long) x;
    const unsigned long xp = (unsigned long)((x + 1) % Nx);
    const unsigned long xm = (unsigned long)((x + Nx - 1) % Nx);
    const unsigned long y0 = (unsigned long) y * (unsigned long) Nx;
    const unsigned long yp = (unsigned long)(((y + 1) % Ny)) * (unsigned long) Nx;
    const unsigned long ym = (unsigned long)(((y + Ny - 1) % Ny)) * (unsigned long) Nx;
    unsigned long j[Q];
    j[0] = x0 + y0; // = n
    j[1] = xp + y0; j[2] = xm + y0;
    j[3] = x0 + yp; j[4] = x0 + ym;
    j[5] = xp + yp; j[6] = xm + ym;
    j[7] = xp + ym; j[8] = xm + yp;

    const bool odd = (t_store % 2ULL) != 0ULL;

    // Invert store_f() (kernel.cpp):
    //   fi[index_f(n, 0)]                       = fhn[0];
    //   fi[index_f(j[i], odd ? i+1 : i  )]      = fhn[i];     (i odd)
    //   fi[index_f(n,    odd ? i   : i+1)]      = fhn[i+1];   (i odd)
    // with index_f(cell, dir) = dir*N + cell.
    fhn_FX[0] = fi[0u * N + j[0]];
    for (int i = 1; i < Q; i += 2) {
        const int dir_i   = odd ? (i + 1) : i;
        const int dir_ip1 = odd ? i : (i + 1);
        fhn_FX[i]     = fi[(unsigned long) dir_i   * N + j[i]];
        fhn_FX[i + 1] = fi[(unsigned long) dir_ip1 * N + j[0]];
    }

    // FluidX3D stores DDFs shifted by the rest weight (perturbation method /
    // DDF-shifting): fi_stored = f_physical - w_i (see calculate_rho_u() in
    // kernel.cpp, which adds 1.0 back to rho). Undo the shift to recover the
    // physical populations. FluidX3D D2Q9 weights (w() in kernel.cpp): index 0
    // = 4/9, axis dirs 1-4 = 1/9, diagonal dirs 5-8 = 1/36.
    static const float w_FX[Q] = {
        4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
    };
    for (int i = 0; i < Q; ++i) fhn_FX[i] += w_FX[i];
}

// ---- V1 hook ----

namespace {

void open_csv_if_needed(V1Hook& hook) {
    if (hook.csv_handle != nullptr) return;
    if (hook.csv_path == nullptr)   return;
    FILE* fp = std::fopen(hook.csv_path, "w");
    if (fp == nullptr) {
        std::fprintf(stderr, "[paper3] ERROR: failed to open '%s' for writing.\n", hook.csv_path);
        return;
    }
    std::fprintf(fp, "step,u_w,tau_plus,Ma,epsilon_g_hat_mean,epsilon_g_hat_std,n_samples,n_corner_excluded,build_hash\n");
    std::fflush(fp);
    hook.csv_handle = (void*) fp;
}

} // anonymous namespace

bool v1_hook_should_sample(const V1Hook& hook, unsigned long long step) {
    if (hook.csv_path == nullptr) return false;
    if (step < hook.sample_step_start) return false;
    return ((step - hook.sample_step_start) % hook.sample_cadence) == 0ULL;
}

namespace {

// One-time validation: reconstruct rho/u at an interior row and compare to
// FluidX3D's own device fields. At interior (non-wall) cells, collision
// conserves mass and momentum, so the post-collision reconstruction must
// reproduce FluidX3D's rho[n] and u[n] (computed by update_fields() through
// the correct Esoteric-Pull indexing). A gross mismatch means the read-back
// indexing or the time-step parity is wrong. Non-circular: the reference is
// FluidX3D's own kernel output, not our synthetic data.
void validate_readback(V1Hook& hook, unsigned long long t_store,
                       const float* fi_raw, unsigned long N,
                       const float* rho_fx, const float* u_fx) {
    const int Nx = hook.Nx, Ny = hook.Ny;
    const int y_val = Ny / 2; // interior row, away from both walls
    const int x_lo = hook.corner_buffer;
    const int x_hi = Nx - 1 - hook.corner_buffer;
    double max_drho = 0.0, max_du = 0.0;
    int n_checked = 0;
    float fhn_FX[Q];
    double f_LL[Q];
    for (int x = x_lo; x <= x_hi; ++x) {
        const unsigned long cell = (unsigned long) x + (unsigned long) y_val * (unsigned long) Nx;
        reconstruct_post_collision_D2Q9(fi_raw, N, x, y_val, Nx, Ny, t_store, fhn_FX);
        fluidx3d_to_LL_D2Q9(fhn_FX, f_LL);
        double rho_r = 0.0, jx_r = 0.0, jy_r = 0.0;
        for (int k = 0; k < Q; ++k) { rho_r += f_LL[k]; jx_r += c_x[k]*f_LL[k]; jy_r += c_y[k]*f_LL[k]; }
        const double ux_r = jx_r / rho_r, uy_r = jy_r / rho_r;
        const double drho = std::fabs(rho_r - (double) rho_fx[cell]);
        const double dux  = std::fabs(ux_r  - (double) u_fx[cell]);
        const double duy  = std::fabs(uy_r  - (double) u_fx[N + cell]);
        if (drho > max_drho) max_drho = drho;
        if (dux  > max_du)   max_du   = dux;
        if (duy  > max_du)   max_du   = duy;
        ++n_checked;
    }
    const double tol = 1e-3; // FP32 populations; this catches indexing/parity errors, not float noise
    const bool ok = (max_drho < tol) && (max_du < tol);
    std::fprintf(stderr,
        "[paper3] read-back validation @ t_store=%llu, interior row y=%d, %d cells: "
        "max|d_rho|=%.3e max|d_u|=%.3e -> %s\n",
        t_store, y_val, n_checked, max_drho, max_du, ok ? "PASS" : "FAIL (indexing/parity bug!)");
    hook.readback_validated = true;
}

} // anonymous namespace

bool v1_hook_tick(V1Hook& hook,
                  unsigned long long step,
                  const float* fi_raw,
                  unsigned long N,
                  const float* rho_fx,
                  const float* u_fx) {
    if (!v1_hook_should_sample(hook, step)) return false;

    open_csv_if_needed(hook);
    if (hook.csv_handle == nullptr) return false;

    const int Nx = hook.Nx;
    const int Ny = hook.Ny;
    const int y  = hook.wall_y;
    const int x_lo = hook.corner_buffer;
    const int x_hi = Nx - 1 - hook.corner_buffer;
    const double cs = std::sqrt(cs2);
    const double Ma = hook.u_w / cs;
    const unsigned long long t_store = step - 1ULL; // t passed to store_f for the data now in fi_raw

    if (y < 0 || y >= Ny) {
        std::fprintf(stderr, "[paper3] WARNING: wall_y=%d out of range [0, %d).\n", y, Ny);
        return false;
    }
    if (x_lo > x_hi) {
        std::fprintf(stderr, "[paper3] WARNING: corner_buffer=%d leaves zero sample cells.\n", hook.corner_buffer);
        return false;
    }

    // One-time read-back validation against FluidX3D's own rho/u fields.
    if (!hook.readback_validated && rho_fx != nullptr && u_fx != nullptr) {
        validate_readback(hook, t_store, fi_raw, N, rho_fx, u_fx);
    }

    // F_pump control-surface flux pipeline (Lock 1 machinery): integrate u . n
    // over the control line x = flux_plane and accumulate the time-average.
    // For V1 (2D Couette) n = x-hat, so Q = sum_y u_x(flux_plane, y).
    if (hook.measure_flux && u_fx != nullptr) {
        const int y_lo = hook.flux_y_lo;
        const int y_hi = (hook.flux_y_hi > 0) ? hook.flux_y_hi : (Ny - 2);
        double Q = 0.0;
        for (int yy = y_lo; yy <= y_hi; ++yy) {
            const unsigned long cell = (unsigned long) hook.flux_plane + (unsigned long) yy * (unsigned long) Nx;
            Q += (double) u_fx[cell]; // u_x component is the first N entries
        }
        hook.flux_Q_sum += Q;
        hook.flux_n_samples += 1u;
    }

    // Accumulate mean / variance over sampled cells via Welford's algorithm.
    double mean = 0.0, M2 = 0.0;
    int n = 0;
    int corner_excluded = 2 * hook.corner_buffer;

    float fhn_FX[Q];
    double f_LL[Q];
    for (int x = x_lo; x <= x_hi; ++x) {
        reconstruct_post_collision_D2Q9(fi_raw, N, x, y, Nx, Ny, t_store, fhn_FX);
        fluidx3d_to_LL_D2Q9(fhn_FX, f_LL);
        const double eg = compute_eps_g_hat_full(f_LL, hook.tau_plus, Ma, hook.kick_rest_reference);

        ++n;
        const double delta  = eg - mean;
        mean += delta / (double) n;
        const double delta2 = eg - mean;
        M2 += delta * delta2;
    }

    const double std_dev = (n > 1) ? std::sqrt(M2 / (double)(n - 1)) : 0.0;

    FILE* fp = (FILE*) hook.csv_handle;
    std::fprintf(fp, "%llu,%.8f,%.8f,%.8f,%.10e,%.10e,%d,%d,%s\n",
                 step, hook.u_w, hook.tau_plus, Ma,
                 mean, std_dev, n, corner_excluded, hook.build_hash);
    std::fflush(fp);

    ++hook.n_samples_taken;
    return true;
}

void v1_hook_close(V1Hook& hook) {
    if (hook.measure_flux && hook.flux_n_samples > 0u) {
        const double Q_mean = hook.flux_Q_sum / (double) hook.flux_n_samples;
        std::fprintf(stderr,
            "[paper3] F_pump pipeline: time-averaged control-surface flux Q = %.6f "
            "over %u samples (x=%d, y=[%d,%d])",
            Q_mean, hook.flux_n_samples, hook.flux_plane, hook.flux_y_lo,
            (hook.flux_y_hi > 0) ? hook.flux_y_hi : -1);
        if (hook.flux_Q_ref != 0.0) {
            std::fprintf(stderr, "; Q_ref=%.6f  Q/Q_ref=%.6f  (rel.err %.3e)",
                hook.flux_Q_ref, Q_mean / hook.flux_Q_ref,
                (Q_mean - hook.flux_Q_ref) / hook.flux_Q_ref);
        }
        std::fprintf(stderr, "\n");
    }
    if (hook.csv_handle == nullptr) return;
    std::fclose((FILE*) hook.csv_handle);
    hook.csv_handle = nullptr;
}

V1Hook g_v1_hook;

} // namespace paper3
