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
                         const int* subspace, int n_subspace) {
    double rho = 0.0, jx = 0.0, jy = 0.0;
    for (int i = 0; i < Q; ++i) {
        rho += f[i];
        jx  += c_x[i] * f[i];
        jy  += c_y[i] * f[i];
    }
    const double ux = jx / rho;
    const double uy = jy / rho;

    double feq[Q];
    compute_feq(rho, ux, uy, feq);

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

double compute_eps_g_hat_full(const double f[Q], double tau_plus, double Ma) {
    return compute_eps_g_hat(f, tau_plus, Ma, NON_CONSERVED, 6);
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

bool v1_hook_tick(V1Hook& hook,
                  unsigned long long step,
                  const float* f_fluidx3d,
                  unsigned long N) {
    if (hook.csv_path == nullptr) return false;

    // Skip until first scheduled sample, then sample every cadence steps.
    if (step < hook.sample_step_start) return false;
    if (((step - hook.sample_step_start) % hook.sample_cadence) != 0ULL) return false;

    open_csv_if_needed(hook);
    if (hook.csv_handle == nullptr) return false;

    const int Nx = hook.Nx;
    const int Ny = hook.Ny;
    const int y  = hook.wall_y;
    const int x_lo = hook.corner_buffer;
    const int x_hi = Nx - 1 - hook.corner_buffer;
    const double cs = std::sqrt(cs2);
    const double Ma = hook.u_w / cs;

    if (y < 0 || y >= Ny) {
        std::fprintf(stderr, "[paper3] WARNING: wall_y=%d out of range [0, %d).\n", y, Ny);
        return false;
    }
    if (x_lo > x_hi) {
        std::fprintf(stderr, "[paper3] WARNING: corner_buffer=%d leaves zero sample cells.\n", hook.corner_buffer);
        return false;
    }

    // Accumulate mean / variance over sampled cells via Welford's algorithm.
    double mean = 0.0, M2 = 0.0;
    int n = 0;
    int corner_excluded = 2 * hook.corner_buffer;

    float f_FX[Q];
    double f_LL[Q];
    for (int x = x_lo; x <= x_hi; ++x) {
        const unsigned long cell = (unsigned long) x + (unsigned long) y * (unsigned long) Nx;
        for (int i = 0; i < Q; ++i) {
            f_FX[i] = f_fluidx3d[(unsigned long) i * N + cell];
        }
        fluidx3d_to_LL_D2Q9(f_FX, f_LL);
        const double eg = compute_eps_g_hat_full(f_LL, hook.tau_plus, Ma);

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
    if (hook.csv_handle == nullptr) return;
    std::fclose((FILE*) hook.csv_handle);
    hook.csv_handle = nullptr;
}

V1Hook g_v1_hook;

} // namespace paper3
