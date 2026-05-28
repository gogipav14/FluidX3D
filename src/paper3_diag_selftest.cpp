// paper3_diag_selftest.cpp -- Paper 3 V1 ghost-energy hook math kernel + self-test.
//
// Standalone test file. Compile and run with:
//   g++ -std=c++17 -O2 src/paper3_diag_selftest.cpp -o paper3_diag_selftest && ./paper3_diag_selftest
//
// Verifies the math kernel that will eventually become paper3_diag.cpp inside
// FluidX3D. This file is NOT linked into the FluidX3D solver; it has its own
// main() and depends only on <cstdio> / <cmath>. Math conventions are locked
// in plans/paper3_d2q9_face_wall_derivation.md.
//
// Expected results:
//   - synthetic Ladd injection on a D2Q9 top face wall, x-motion -> eps_g_hat = 5/18
//   - velocity-independence sweep                                -> eps_g_hat = 5/18 across u_w
//   - tau_plus sweep                                             -> eps_g_hat = 5/18 across tau_plus
//   - "higher-order-only" subspace bug control                   -> 1/36 (NOT 5/18)
//   - equilibrium-only input (no ghost perturbation)             -> 0

#include <cstdio>
#include <cmath>
#include <cstdlib>

namespace paper3 {

constexpr int Q = 9;
constexpr double cs2 = 1.0 / 3.0;

// Direction ordering (Lallemand-Luo 2000):
//   0 = (0,0), 1 = (+x), 2 = (+y), 3 = (-x), 4 = (-y),
//   5 = (+x,+y), 6 = (-x,+y), 7 = (-x,-y), 8 = (+x,-y)
constexpr double c_x[Q] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
constexpr double c_y[Q] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
constexpr double w_i[Q] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                           1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

// d'Humieres moment basis (Lallemand-Luo 2000, Table I).
// Row index k = mode; column index i = direction.
// Modes: 0=rho, 1=e, 2=eps, 3=jx, 4=qx, 5=jy, 6=qy, 7=pxx, 8=pxy
constexpr int h[Q][Q] = {
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
constexpr int CONSERVED[3] = {0, 3, 5};
constexpr int NON_CONSERVED[6] = {1, 2, 4, 6, 7, 8};
constexpr int NARROW_GHOST[4] = {1, 2, 4, 6};  // e, eps, q_x, q_y (subspace-bug control)

constexpr bool is_conserved(int k) { return k == 0 || k == 3 || k == 5; }

// Weighted moment norm ||h_k||_w^2 = sum_i w_i h[k][i]^2
double norm_w_sq(int k) {
    double s = 0.0;
    for (int i = 0; i < Q; ++i) s += w_i[i] * h[k][i] * h[k][i];
    return s;
}

// Unweighted moment norm ||h_k||^2 = sum_i h[k][i]^2 (Lallemand-Luo orthogonality)
double norm_sq(int k) {
    double s = 0.0;
    for (int i = 0; i < Q; ++i) s += (double)h[k][i] * h[k][i];
    return s;
}

// Standard D2Q9 equilibrium, second order in u.
void compute_feq(double rho, double ux, double uy, double feq[Q]) {
    const double u2 = ux * ux + uy * uy;
    for (int i = 0; i < Q; ++i) {
        const double cu = c_x[i] * ux + c_y[i] * uy;
        feq[i] = w_i[i] * rho * (1.0 + cu / cs2 + 0.5 * cu * cu / (cs2 * cs2) - 0.5 * u2 / cs2);
    }
}

// Forward moment transform: m_k = sum_i h[k][i] f_i
void moment_transform(const double f[Q], double m[Q]) {
    for (int k = 0; k < Q; ++k) {
        double s = 0.0;
        for (int i = 0; i < Q; ++i) s += h[k][i] * f[i];
        m[k] = s;
    }
}

// Full V1-gate pipeline:
//   1. Conserved moments rho, j_x, j_y from f.
//   2. f_neq = f - f_eq(rho, u).
//   3. Moments of f_neq (conserved moments are zero up to roundoff).
//   4. eps_g = (1/(2 c_s^2)) * (1/2) * sum_{k in N} m_k^2 / ||h_k||_w^2
//   5. eps_g_hat = eps_g / (tau^2 Ma^2)
//
// The (1/(2 c_s^2)) prefactor matches Paper 2 Phase A Lock 3 -- WITHOUT it the
// V1 gate lands at 5/27 instead of 5/18 (off by a factor 3/2). See locked
// derivation in plans/paper3_d2q9_face_wall_derivation.md.
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

// Convenience wrapper using the full non-conserved subspace (the V1 gate definition).
double compute_eps_g_hat_full(const double f[Q], double tau_plus, double Ma) {
    return compute_eps_g_hat(f, tau_plus, Ma, NON_CONSERVED, 6);
}

// Build synthetic post-collision populations representing the steady-state
// Ladd-injected ghost at a top wall with u_w = (u_w, 0):
//   m_{q_x}^ss  = +tau_plus * u_w / 3
//   m_{p_xy}^ss = -tau_plus * u_w / 3
// (Other non-conserved modes have zero injection for this geometry; see
// plans/paper3_d2q9_face_wall_derivation.md §1.4 table.)
//
// Base state: rho = 1, u = 0 (a synthetic rest cell -- not the real V1 wall-
// adjacent flow profile, but isolating the kernel from the macroscopic flow).
// f_synth = f^eq(1, 0, 0) + sum_{k = q_x, p_xy} m_k^ss * h[k][i] / ||h_k||^2_unweighted.
void build_synthetic_ladd_top_wall_x(double u_w, double tau_plus, double f_synth[Q]) {
    double feq[Q];
    compute_feq(1.0, 0.0, 0.0, feq);

    const double m_qx_ss  = +tau_plus * u_w / 3.0;
    const double m_pxy_ss = -tau_plus * u_w / 3.0;
    const double N_qx  = norm_sq(4);  // = 12
    const double N_pxy = norm_sq(8);  // = 4

    for (int i = 0; i < Q; ++i) {
        f_synth[i] = feq[i]
                   + h[4][i] * m_qx_ss  / N_qx
                   + h[8][i] * m_pxy_ss / N_pxy;
    }
}

} // namespace paper3

// ---------------------------------------------------------------------------

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

    // ---- Test 0: basis orthogonality sanity (unweighted) -----------------------
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

    // ---- Test 1: pure equilibrium has zero ghost energy ------------------------
    {
        std::printf("[Test 1] Pure equilibrium (no ghost perturbation) -> eps_g_hat = 0:\n");
        double f[Q];
        compute_feq(1.0, 0.04, 0.0, f);
        const double Ma = 0.04 / cs;
        const double eg = compute_eps_g_hat_full(f, 0.55, Ma);
        check("rho=1, ux=0.04, equilibrium only", eg, 0.0, 1e-12);
        std::printf("\n");
    }

    // ---- Test 2: synthetic Ladd injection -> 5/18 -----------------------------
    {
        std::printf("[Test 2] Synthetic Ladd injection -> eps_g_hat = 5/18:\n");
        double f[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f);
        const double Ma = 0.04 / cs;
        const double eg = compute_eps_g_hat_full(f, 0.55, Ma);
        check("u_w=0.04, tau=0.55, full non-conserved (6 modes)", eg, TARGET, 1e-10);
        std::printf("\n");
    }

    // ---- Test 3: velocity-independence sweep ---------------------------------
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

    // ---- Test 4: tau_plus-independence sweep ---------------------------------
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

    // ---- Test 5: subspace-bug control ----------------------------------------
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

    // ---- Test 6: dependence on which modes contribute -----------------------
    //
    //   For top wall + x-motion, only q_x and p_xy carry non-zero steady-state
    //   ghost moments. Verify each independently.
    {
        std::printf("[Test 6] Per-mode contribution at u_w=0.04, tau=0.55:\n");
        double f[Q];
        build_synthetic_ladd_top_wall_x(0.04, 0.55, f);

        // Compute m_k from (f - f_eq) and report.
        double rho = 0, jx = 0, jy = 0;
        for (int i = 0; i < Q; ++i) { rho += f[i]; jx += c_x[i]*f[i]; jy += c_y[i]*f[i]; }
        double feq[Q]; compute_feq(rho, jx/rho, jy/rho, feq);
        double f_neq[Q]; for (int i = 0; i < Q; ++i) f_neq[i] = f[i] - feq[i];
        double m[Q]; moment_transform(f_neq, m);

        // Expected values per derivation §1.5:
        // m_q_x  = +tau u_w/3 = 0.55 * 0.04 / 3 = 0.00733...
        // m_p_xy = -0.00733...
        // All other non-conserved modes ~ 0 to machine precision.
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
