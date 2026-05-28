// paper3_diag_selftest.cpp -- Standalone self-test for the V1 ghost-energy
// hook math kernel defined in paper3_diag.{hpp,cpp}.
//
// Compile and run with:
//   g++ -std=c++17 -O2 src/paper3_diag.cpp src/paper3_diag_selftest.cpp \
//       -o paper3_diag_selftest && ./paper3_diag_selftest
//
// Expected output: 20 PASS, 0 FAIL. eps_g_hat target = 5/18 to machine
// precision under synthetic Ladd top-wall x-motion injection.
//
// Conventions and target derivation: plans/paper3_d2q9_face_wall_derivation.md.

#include "paper3_diag.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>

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

    std::printf("=== Summary: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
