// paper3_diag_selftest.cpp -- Standalone self-test for the V1 ghost-energy
// hook math kernel defined in paper3_diag.{hpp,cpp}.
//
// Compile and run with:
//   g++ -std=c++17 -O2 -DPAPER3_DIAG_SELFTEST_MAIN src/paper3_diag.cpp src/paper3_diag_selftest.cpp -o paper3_diag_selftest && ./paper3_diag_selftest
//
// The -DPAPER3_DIAG_SELFTEST_MAIN guard is required: without it the file is
// empty, which is the desired behaviour for make.sh / the FluidX3D build that
// globs src/*.cpp (FluidX3D already has its own main() in main.cpp).
//
// Expected output: 34 PASS, 0 FAIL. eps_g_hat target = 5/18 to machine
// precision under synthetic Ladd top-wall x-motion injection. Tests 7-10
// additionally exercise the FluidX3D-ordering translation, the Esoteric-Pull
// post-collision reconstruction (both parities), and the V1Hook CSV pipeline.
//
// Conventions and target derivation: plans/paper3_d2q9_face_wall_derivation.md.

#ifdef PAPER3_DIAG_SELFTEST_MAIN

#include "paper3_diag.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <initializer_list>

namespace {

int n_pass = 0;
int n_fail = 0;

void check(const char* name, double val, double expected, double tol) {
    const double err = std::abs(val - expected);
    const bool ok = err < tol;
    std::printf("  %-58s = %.10f  (target %.10f, err %.2e)  %s\n",
                name, val, expected, err, ok ? "PASS" : "FAIL");
    if (ok) ++n_pass; else ++n_fail;
}

// Mirror store_f() (kernel.cpp) to build a raw Esoteric-Pull device buffer from
// a spatially uniform post-collision field fhn_FX (FluidX3D direction order),
// for the given time-step parity. Because the field is uniform, every slot
// receives the same component no matter which cell wrote it, so
// reconstruct_post_collision_D2Q9() must recover fhn_FX at any cell. This lets
// the offline self-test exercise the exact indexing path the live hook uses.
void store_f_scatter_uniform(float* raw, unsigned long N, int Nx, int Ny,
                             unsigned long long t_store, const float fhn_FX_physical[paper3::Q]) {
    const int Q = paper3::Q;
    const bool odd = (t_store % 2ULL) != 0ULL;
    // Mirror FluidX3D's DDF-shifting: store f_physical - w_i (so reconstruct's
    // +w_i recovers the physical value). FluidX3D D2Q9 weights by index.
    static const float w_FX[9] = {
        4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
    };
    float fhn_FX[paper3::Q];
    for (int i = 0; i < Q; ++i) fhn_FX[i] = fhn_FX_physical[i] - w_FX[i];
    for (int y = 0; y < Ny; ++y) for (int x = 0; x < Nx; ++x) {
        const unsigned long x0 = (unsigned long) x;
        const unsigned long xp = (unsigned long)((x + 1) % Nx);
        const unsigned long xm = (unsigned long)((x + Nx - 1) % Nx);
        const unsigned long y0 = (unsigned long) y * (unsigned long) Nx;
        const unsigned long yp = (unsigned long)(((y + 1) % Ny)) * (unsigned long) Nx;
        const unsigned long ym = (unsigned long)(((y + Ny - 1) % Ny)) * (unsigned long) Nx;
        unsigned long j[9];
        j[0]=x0+y0; j[1]=xp+y0; j[2]=xm+y0; j[3]=x0+yp; j[4]=x0+ym;
        j[5]=xp+yp; j[6]=xm+ym; j[7]=xp+ym; j[8]=xm+yp;
        raw[0u*N + j[0]] = fhn_FX[0];
        for (int i = 1; i < Q; i += 2) {
            raw[(unsigned long)(odd ? i+1 : i  )*N + j[i]] = fhn_FX[i];
            raw[(unsigned long)(odd ? i   : i+1)*N + j[0]] = fhn_FX[i+1];
        }
    }
}

} // anonymous namespace

int main() {
    using namespace paper3;

    const double cs = std::sqrt(cs2);
    const double TARGET = 5.0 / 18.0;
    const double BUG_TARGET = 1.0 / 36.0;

    std::printf("=== paper3_diag self-test (D2Q9 face-wall Ladd injection) ===\n");
    std::printf("Lattice: D2Q9, c_s^2 = 1/3.\n");
    std::printf("Basis:   Lallemand-Luo 2000 d'Humieres (see paper3_d2q9_face_wall_derivation.md).\n");
    std::printf("Energy:  eps_g^bnd = (1/(2 c_s^2)) * (1/2) * sum_{k in N} m_k^2 / ||h_k||_w^2.\n");
    std::printf("V1 gate target:  eps_g_hat = 5/18 = %.10f\n", TARGET);
    std::printf("Subspace-bug:    eps_g_hat = 1/36 = %.10f (dropped stress modes)\n\n", BUG_TARGET);

    // ---- Test 0: basis orthogonality sanity (unweighted) ----
    std::printf("[Test 0] Lallemand-Luo basis orthogonality (unweighted):\n");
    bool ortho_ok = true;
    for (int k = 0; k < Q && ortho_ok; ++k) {
        for (int l = 0; l < Q && ortho_ok; ++l) {
            int s = 0;
            for (int i = 0; i < Q; ++i) s += h[k][i] * h[l][i];
            const int nk = (int) norm_sq(k);
            if (k == l) { if (s != nk) ortho_ok = false; }
            else        { if (s != 0)  ortho_ok = false; }
        }
    }
    std::printf("  %-58s = %s\n", "Orthogonality of all 9 basis vectors", ortho_ok ? "PASS" : "FAIL");
    if (ortho_ok) ++n_pass; else ++n_fail;

    std::printf("[Test 0] Weighted norms ||h_k||_w^2 for the 6 non-conserved modes:\n");
    const char* names[Q] = {"rho","e","eps","j_x","q_x","j_y","q_y","p_xx","p_xy"};
    for (int k : NON_CONSERVED) {
        std::printf("  ||%s||_w^2 = %.6f\n", names[k], norm_w_sq(k));
    }
    std::printf("\n");

    // ---- Test 1: pure equilibrium has zero ghost energy ----
    {
        std::printf("[Test 1] Pure equilibrium (no ghost perturbation) -> eps_g_hat = 0:\n");
        double f[Q];
        compute_feq(1.0, 0.04, 0.0, f);
        const double Ma = 0.04 / cs;
        const double eg = compute_eps_g_hat_full(f, 0.55, Ma);
        check("rho=1, ux=0.04, equilibrium only", eg, 0.0, 1e-12);
        std::printf("\n");
    }

    // ---- Test 2: synthetic Ladd injection -> 5/18 ----
    {
        std::printf("[Test 2] Synthetic Ladd injection -> eps_g_hat = 5/18:\n");
        double f[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f);
        const double Ma = 0.04 / cs;
        const double eg = compute_eps_g_hat_full(f, 0.55, Ma);
        check("u_w=0.04, tau=0.55, full non-conserved (6 modes)", eg, TARGET, 1e-10);
        std::printf("\n");
    }

    // ---- Test 3: velocity-independence sweep ----
    {
        std::printf("[Test 3] Velocity-independence: eps_g_hat constant across u_w sweep:\n");
        const double uw_vals[] = {0.02, 0.04, 0.08};
        double results[3];
        for (int j = 0; j < 3; ++j) {
            double f[Q];
            build_synthetic_ladd_top_wall_x(uw_vals[j], 0.55, f);
            const double Ma = uw_vals[j] / cs;
            results[j] = compute_eps_g_hat_full(f, 0.55, Ma);
            char name[64];
            std::snprintf(name, sizeof(name), "u_w=%.2f -> 5/18", uw_vals[j]);
            check(name, results[j], TARGET, 1e-10);
        }
        const double rel_spread = std::fabs(results[2] - results[0]) / results[0];
        check("relative spread |max-min|/value < 1%", rel_spread, 0.0, 1e-10);
        std::printf("\n");
    }

    // ---- Test 4: tau_plus-independence sweep ----
    {
        std::printf("[Test 4] tau_plus-independence: eps_g_hat constant across tau sweep:\n");
        const double tau_vals[] = {0.51, 0.55, 0.65, 0.80, 1.00};
        for (int j = 0; j < 5; ++j) {
            double f[Q];
            build_synthetic_ladd_top_wall_x(0.04, tau_vals[j], f);
            const double Ma = 0.04 / cs;
            const double eg = compute_eps_g_hat_full(f, tau_vals[j], Ma);
            char name[64];
            std::snprintf(name, sizeof(name), "tau_plus=%.2f -> 5/18", tau_vals[j]);
            check(name, eg, TARGET, 1e-10);
        }
        std::printf("\n");
    }

    // ---- Test 5: subspace-bug control ----
    {
        std::printf("[Test 5] Subspace-bug control (drop stress modes p_xx, p_xy):\n");
        std::printf("         A diagnostic that sums only {e, eps, q_x, q_y} -- the\n");
        std::printf("         Lallemand-Luo narrow-sense ghost -- lands at 1/36, NOT\n");
        std::printf("         at 5/18. Both numbers below must match expected.\n");
        double f[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f);
        const double Ma = 0.04 / cs;
        const double eg_correct = compute_eps_g_hat(f, 0.55, Ma, NON_CONSERVED, 6);
        const double eg_buggy   = compute_eps_g_hat(f, 0.55, Ma, NARROW_GHOST, 4);
        check("CORRECT subspace (all 6 non-conserved) -> 5/18", eg_correct, TARGET, 1e-10);
        check("BUGGY  subspace (higher-order only)    -> 1/36", eg_buggy,   BUG_TARGET, 1e-10);
        std::printf("\n");
    }

    // ---- Test 6: per-mode contribution breakdown ----
    {
        std::printf("[Test 6] Per-mode contribution at u_w=0.04, tau=0.55:\n");
        double f[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f);

        double rho = 0, jx = 0, jy = 0;
        for (int i = 0; i < Q; ++i) { rho += f[i]; jx += c_x[i]*f[i]; jy += c_y[i]*f[i]; }
        double feq[Q]; compute_feq(rho, jx/rho, jy/rho, feq);
        double f_neq[Q]; for (int i = 0; i < Q; ++i) f_neq[i] = f[i] - feq[i];
        double m[Q]; moment_transform(f_neq, m);

        std::printf("    m_e     = %+12.6e   (expected ~0)\n", m[1]);
        std::printf("    m_eps   = %+12.6e   (expected ~0)\n", m[2]);
        std::printf("    m_qx    = %+12.6e   (expected +tau*u_w/3 = %+12.6e)\n",
                    m[4], +0.55 * 0.04 / 3.0);
        std::printf("    m_qy    = %+12.6e   (expected ~0)\n", m[6]);
        std::printf("    m_pxx   = %+12.6e   (expected ~0)\n", m[7]);
        std::printf("    m_pxy   = %+12.6e   (expected -tau*u_w/3 = %+12.6e)\n",
                    m[8], -0.55 * 0.04 / 3.0);

        const double expected_qx  = +0.55 * 0.04 / 3.0;
        const double expected_pxy = -0.55 * 0.04 / 3.0;
        check("m_q_x   matches +tau u_w/3",      m[4], expected_qx,  1e-12);
        check("m_p_xy  matches -tau u_w/3",      m[8], expected_pxy, 1e-12);
        check("m_e     ~ 0 (no e injection)",    m[1], 0.0, 1e-12);
        check("m_eps   ~ 0 (no eps injection)",  m[2], 0.0, 1e-12);
        check("m_q_y   ~ 0 (no q_y injection)",  m[6], 0.0, 1e-12);
        check("m_p_xx  ~ 0 (no p_xx injection)", m[7], 0.0, 1e-12);
        std::printf("\n");
    }

    // ---- Test 7: FluidX3D <-> Lallemand-Luo translation roundtrip ----
    {
        std::printf("[Test 7] FluidX3D <-> Lallemand-Luo translation roundtrip:\n");
        double f_LL[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f_LL);
        float f_FX[Q];
        LL_to_fluidx3d_D2Q9(f_LL, f_FX);
        double f_LL_back[Q];
        fluidx3d_to_LL_D2Q9(f_FX, f_LL_back);

        double max_err = 0.0;
        for (int i = 0; i < Q; ++i) {
            const double err = std::abs(f_LL[i] - f_LL_back[i]);
            if (err > max_err) max_err = err;
        }
        check("roundtrip max |f_LL - f_LL_back| ~ 0", max_err, 0.0, 1e-6);

        // Eps_g after roundtrip must still land at 5/18.
        const double Ma = 0.04 / cs;
        const double eg = compute_eps_g_hat_full(f_LL_back, 0.55, Ma);
        check("eps_g_hat after roundtrip = 5/18", eg, TARGET, 1e-6);

        // Translation table direction check: c_x[ll_to_fx[k]] in FX ordering
        // should equal c_x[k] in LL ordering (velocity is preserved).
        constexpr double c_x_FX[Q] = { 0,  1, -1,  0,  0,  1, -1,  1, -1};
        constexpr double c_y_FX[Q] = { 0,  0,  0,  1, -1,  1, -1, -1,  1};
        bool dirs_ok = true;
        for (int k = 0; k < Q; ++k) {
            const int fx_idx = ll_to_fx[k];
            if (c_x_FX[fx_idx] != c_x[k]) dirs_ok = false;
            if (c_y_FX[fx_idx] != c_y[k]) dirs_ok = false;
        }
        std::printf("  %-58s = %s\n", "ll_to_fx table preserves velocities", dirs_ok ? "PASS" : "FAIL");
        if (dirs_ok) ++n_pass; else ++n_fail;
        std::printf("\n");
    }

    // ---- Test 8: Esoteric-Pull reconstruction roundtrip (both parities) ----
    //
    // Scatter a known post-collision field into a raw buffer exactly as
    // store_f() does, then reconstruct it with reconstruct_post_collision_D2Q9()
    // and confirm recovery. This directly tests the indexing/parity logic that
    // the live hook relies on -- the part the old naive read silently got wrong.
    {
        std::printf("[Test 8] Esoteric-Pull store_f/reconstruct roundtrip:\n");
        const int Nx_test = 16, Ny_test = 8;
        const unsigned long N_test = (unsigned long) Nx_test * Ny_test;

        double f_LL[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f_LL);
        float fhn_FX[Q];
        LL_to_fluidx3d_D2Q9(f_LL, fhn_FX);

        for (unsigned long long t_store : {0ULL, 1ULL}) { // even and odd parity
            float* raw = new float[Q * N_test];
            for (unsigned long k = 0; k < (unsigned long)Q * N_test; ++k) raw[k] = -999.0f; // poison
            store_f_scatter_uniform(raw, N_test, Nx_test, Ny_test, t_store, fhn_FX);

            // Reconstruct at an interior cell and a wall-adjacent cell.
            double max_err = 0.0;
            for (int y : {Ny_test/2, Ny_test-2}) {
                for (int x : {5, 8, 11}) {
                    float rec[Q];
                    reconstruct_post_collision_D2Q9(raw, N_test, x, y, Nx_test, Ny_test, t_store, rec);
                    for (int i = 0; i < Q; ++i) {
                        const double err = std::abs((double)rec[i] - (double)fhn_FX[i]);
                        if (err > max_err) max_err = err;
                    }
                }
            }
            char name[80];
            std::snprintf(name, sizeof(name), "t_store parity %llu: max|rec - fhn| ~ 0", t_store);
            check(name, max_err, 0.0, 1e-6);
            delete[] raw;
        }
        std::printf("\n");
    }

    // ---- Test 9: v1_hook_tick on a real Esoteric-Pull buffer -> 5/18 ----
    //
    // Build the raw device buffer via store_f scatter (not naive layout), drive
    // the cadence logic, and confirm the hook reconstructs and reports 5/18.
    // Also exercises the rho/u read-back validation path (interior reconstructs
    // to rho=1, u=0 for this momentum-neutral synthetic field).
    {
        std::printf("[Test 9] v1_hook_tick on Esoteric-Pull buffer -> 5/18:\n");
        const int Nx_test = 20;
        const int Ny_test = 4;
        const int wall_y  = 2;
        const unsigned long N_test = (unsigned long) Nx_test * Ny_test;

        double f_LL[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f_LL);
        float fhn_FX[Q];
        LL_to_fluidx3d_D2Q9(f_LL, fhn_FX);

        // Sample steps will be 100 and 150 -> t_store = 99, 149 (both odd).
        float* raw = new float[Q * N_test];
        store_f_scatter_uniform(raw, N_test, Nx_test, Ny_test, /*t_store parity*/99ULL, fhn_FX);

        // FluidX3D reference fields: this synthetic field is momentum-neutral, so
        // rho=1, u=0 everywhere. Reconstruction must match -> validation PASS.
        float* rho_fx = new float[N_test];
        float* u_fx   = new float[3 * N_test];
        for (unsigned long n = 0; n < N_test; ++n) { rho_fx[n] = 1.0f; u_fx[n] = 0.0f; u_fx[N_test+n] = 0.0f; u_fx[2*N_test+n] = 0.0f; }

        V1Hook hook;
        const char* csv_path = "paper3_diag_selftest_test9.csv";
        hook.csv_path = csv_path;
        hook.Nx = Nx_test;
        hook.Ny = Ny_test;
        hook.wall_y = wall_y;
        hook.corner_buffer = 4;
        hook.u_w = 0.04;
        hook.tau_plus = 0.55;
        hook.sample_step_start = 100ULL;
        hook.sample_cadence    = 50ULL;
        hook.build_hash = "test9";

        bool took0 = v1_hook_tick(hook,  50ULL, raw, N_test, rho_fx, u_fx);
        bool took1 = v1_hook_tick(hook, 100ULL, raw, N_test, rho_fx, u_fx);
        bool took2 = v1_hook_tick(hook, 150ULL, raw, N_test, rho_fx, u_fx);
        bool took3 = v1_hook_tick(hook, 130ULL, raw, N_test, rho_fx, u_fx); // off-cadence
        v1_hook_close(hook);

        check("step=50 < start: no sample taken",         (double)took0, 0.0, 1e-12);
        check("step=100 (= start): sample taken",         (double)took1, 1.0, 1e-12);
        check("step=150 (start + cadence): sample taken", (double)took2, 1.0, 1e-12);
        check("step=130 (off-cadence): no sample taken",  (double)took3, 0.0, 1e-12);

        FILE* fp = std::fopen(csv_path, "r");
        if (fp == nullptr) {
            std::printf("  FAIL: could not open %s for verification\n", csv_path);
            ++n_fail;
        } else {
            char line[512];
            if (!std::fgets(line, sizeof(line), fp)) {} // header
            if (!std::fgets(line, sizeof(line), fp)) {} // first data row
            double step_v, uw_v, tau_v, ma_v, mean_v, std_v;
            int n_samp_v, n_corner_v;
            char hash_buf[64];
            const int got = std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%s",
                                        &step_v, &uw_v, &tau_v, &ma_v,
                                        &mean_v, &std_v, &n_samp_v, &n_corner_v, hash_buf);
            std::fclose(fp);
            std::remove(csv_path);
            if (got >= 6) {
                check("CSV row 1 epsilon_g_hat_mean = 5/18", mean_v, TARGET, 1e-6);
                check("CSV row 1 epsilon_g_hat_std ~ 0",     std_v,  0.0,    1e-6);
                check("CSV row 1 n_samples = Nx - 2*buffer",
                      (double) n_samp_v, (double)(Nx_test - 2*hook.corner_buffer), 1e-12);
            } else {
                std::printf("  FAIL: CSV parse error (got %d fields)\n", got);
                ++n_fail;
            }
        }

        delete[] raw; delete[] rho_fx; delete[] u_fx;
        std::printf("\n");
    }

    // ---- Test 10: v1_hook_tick on stationary box (no wall motion) -> 0 ----
    {
        std::printf("[Test 10] v1_hook_tick on stationary box (rho=1, u=0) -> 0:\n");
        const int Nx_test = 20;
        const int Ny_test = 4;
        const unsigned long N_test = (unsigned long) Nx_test * Ny_test;

        double feq_LL[Q];
        compute_feq(1.0, 0.0, 0.0, feq_LL);
        float feq_FX[Q];
        LL_to_fluidx3d_D2Q9(feq_LL, feq_FX);

        // Sample at step=100 -> t_store=99 (odd); scatter at matching parity.
        float* raw = new float[Q * N_test];
        store_f_scatter_uniform(raw, N_test, Nx_test, Ny_test, /*t_store parity*/99ULL, feq_FX);

        V1Hook hook;
        const char* csv_path = "paper3_diag_selftest_test10.csv";
        hook.csv_path = csv_path;
        hook.Nx = Nx_test;
        hook.Ny = Ny_test;
        hook.wall_y = 2;
        hook.corner_buffer = 4;
        hook.u_w = 0.04;
        hook.tau_plus = 0.55;
        hook.sample_step_start = 100ULL;
        hook.sample_cadence    = 50ULL;
        hook.build_hash = "test10";

        v1_hook_tick(hook, 100ULL, raw, N_test);
        v1_hook_close(hook);

        FILE* fp = std::fopen(csv_path, "r");
        if (fp != nullptr) {
            char line[512];
            if (!std::fgets(line, sizeof(line), fp)) {} // header
            if (!std::fgets(line, sizeof(line), fp)) {} // data
            double step_v, uw_v, tau_v, ma_v, mean_v, std_v;
            int n_samp_v, n_corner_v;
            char hash_buf[64];
            std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%d,%d,%s",
                        &step_v, &uw_v, &tau_v, &ma_v, &mean_v, &std_v,
                        &n_samp_v, &n_corner_v, hash_buf);
            std::fclose(fp);
            std::remove(csv_path);
            check("stationary box: epsilon_g_hat_mean ~ 0", mean_v, 0.0, 1e-12);
            check("stationary box: epsilon_g_hat_std ~ 0",  std_v,  0.0, 1e-12);
        } else {
            std::printf("  FAIL: could not open %s\n", csv_path);
            ++n_fail;
        }

        delete[] raw;
        std::printf("\n");
    }

    std::printf("=== Summary: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}

#endif // PAPER3_DIAG_SELFTEST_MAIN
