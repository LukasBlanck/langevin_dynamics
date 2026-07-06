// file for evaluating potentials V(r) and derivatives dV/dq
#include <cmath>

// FPU Potential
inline double V_FPU(double r, double w, double beta) {
    return 0.5 * w * w * r * r + 0.25 * beta * r * r * r * r;
}

// FPU derivative
inline double dV_FPU(double r, double w, double beta) {
    return w * w * r + beta * r * r * r;
}


// Josephson Potential
inline double V_J(double r, double EJ) {
    return -EJ * std::cos(r);
}

// Josephson derivative
inline double dV_J(double r, double EJ) {
    return EJ * std::sin(r);
}