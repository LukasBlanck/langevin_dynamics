#pragma once

#include <string>

struct Config {
    struct Grid {
        int N = 64;
    };

    struct Conventions {
        double m = 1.0;
        double kB = 1.0;
    };

    struct Time {
        int N = 10000;
        double end_time = 5.0;
        int save_every = 10;
    };

    struct Ensemble {
        int N = 1000;
    };

    struct Model {
        double omega = 1.0;
        double beta = 1.0;
        double EJ = 1.0;
        double lambda = 1.0;
        double left_bath_T = 2.0;
        std::string potential = "FPU";
    };

    // initialize
    Grid grid;
    Conventions conventions;
    Time time;
    Ensemble ensemble;
    Model model;
};

Config load_config(const std::string &filename);
void print_config(const Config& config);