#pragma once

#include "input/input.hpp"

inline Config create_config() {
    Config config;

    // standard params for testing
    config.grid.N = 128;

    config.conventions.m = 1.0;
    config.conventions.kB = 1.0;

    config.time.end_time = 5.0;
    config.time.N = 100000;
    config.time.save_every = 10;

    config.ensemble.N = 10;

    config.model.beta = 1.0;
    config.model.EJ = 1.0;
    config.model.lambda = 1.0;
    config.model.left_bath_T = 2.0;
    config.model.omega = 1.0;
    config.model.potential = "FPU";

    return config;
}