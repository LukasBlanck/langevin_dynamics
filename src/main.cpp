
#include "BAOAB.hpp"
#include "input/input.hpp"
#include <vector>

// create result arrays -> ej, pearson_corr

// ensemble

// at extract times - write current state into result arrays

int main() {

    // load input params
    const Config config = load_config("src/input/input.toml");
    print_config(config);

    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;
    const double kB = config.conventions.kB;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const int save_every = config.time.save_every;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    const double w = config.model.omega;
    const double beta = config.model.beta;
    const double lambda = config.model.lambda;
    const double left_bath_T = config.model.left_bath_T;
    const double gamma = lambda / m;

    // initialize grid/arrays

    std::vector<double> q(N);
    std::vector<double> p(N);
    std::vector<double> F(N);

    const int n_save = 1 + (N_time + save_every - 1) / save_every;
    std::vector<double> e(n_save * N, 0.0);
    std::vector<double> time(n_save, 0.0);

    int seed = 67;
    std::mt19937_64 rng(seed);

    // initialize with q=p=F=0 for first try
    std::fill(q.begin(), q.end(), 0.0);
    std::fill(p.begin(), p.end(), 0.0);

    // integrate
    BAOAB integrator(config, q, p, F, dt);

    // save initial condition
    int count = 0;
    for (int k = 0; k < N_time; k++) {
        integrator.step_fpu(rng);

        if ((k + 1) % save_every == 0 || k + 1 == N_time) {
            double t = (k + 1) * dt; // time after step
            // save observables
            for (int i = 1; i < N - 1; i++) {
                e[i + count * N] = p[i] * p[i] / (2 * m) + 0.5 * V_FPU(q[i - 1] - q[i], w, beta) +
                                   0.5 * V_FPU(q[i] - q[i + 1], w, beta);
            }
            e[0 + count * N] = p[0] * p[0] / (2 * m) + 0.5 * V_FPU(q[0] - q[1], w, beta);
            e[N - 1 + count * N] =
                p[N - 1] * p[N - 1] / (2 * m) + 0.5 * V_FPU(q[N - 2] - q[N - 1], w, beta);
            time[count] = t;    // store exact time seperately
            count++;
        }
    }

    return 0;
}