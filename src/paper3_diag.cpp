// paper3_diag.cpp -- Implementation of paper3_diag.hpp.
//
// Math conventions: plans/paper3_d2q9_face_wall_derivation.md
// V1 usage:         plans/paper3_phase_B_v1_checklist.md

#include "paper3_diag.hpp"

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

} // namespace paper3
