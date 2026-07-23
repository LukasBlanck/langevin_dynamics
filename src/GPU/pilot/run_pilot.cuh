// this file will run before the demanded production run:
// if FPU requested:
    // run a dedicated GPU/CPU implementation that monitors r_max with
    // demanded dt and returns observables
    // check if dt <= stability bound
    // if yes:
        // 1. run dedicated GPU version with dt/2 and dt/4 that returns observables
        // check if observables stochastically match: result array [N] should match with 1%? to other dt/2.. results:
        // with batch_mean and overall mean and sigma
        // if stochastic error smaller then threshold:
            // check if time resolved:
            // if yes:
                // check if convergence visible
                // if yes:
                    // perfect
                // else:
                    // either increase N_t or N_ensemble
            // esle: run_time_error("N_t too small")
        // esle: run_time_error("N_ensemble too small")
        
        // if yes:
            // run production GPU with dt
        // else:
            // run dt/8 and so on...
    // else: 
        // check stability bound for dt/2
// else if Josephson:
    // check if dt <= stability bound
    // if yes:
        // 1. run dedicated GPU version with dt/2 that returns observables
        // check if observables match and second order convergence
        // estimate error (in what decimal, or how big is stochastic error - possible?)
        // if yes:
            // run production GPU with dt
        // else:
            // run dt/4 and so on...
    // else: 
        // check stability bound for dt/2

#pragma once


#include "input/input.hpp"
inline bool run_pilot(Config &config) {
    
}