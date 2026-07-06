// config.cpp
#include "input.hpp"
#include <iostream>
#include <iomanip>

#include "../../extern/toml.hpp"

Config load_config(const std::string &filename) {
    auto table = toml::parse_file(filename);

    Config config;

    config.grid.N = table["grid"]["N"].value_or(config.grid.N);

    config.conventions.m = table["conventions"]["m"].value_or(config.conventions.m);
    config.conventions.kB = table["conventions"]["kB"].value_or(config.conventions.kB);

    config.time.N = table["time"]["Nt"].value_or(config.time.N);
    config.time.end_time = table["time"]["end_time"].value_or(config.time.end_time);
    config.time.save_every = table["time"]["save_every"].value_or(config.time.save_every);

    config.ensemble.N = table["ensemble"]["N"].value_or(config.ensemble.N);

    config.model.omega = table["model"]["omega"].value_or(config.model.omega);
    config.model.beta = table["model"]["beta"].value_or(config.model.beta);
    config.model.EJ = table["model"]["EJ"].value_or(config.model.EJ);
    config.model.gamma = table["model"]["gamma"].value_or(config.model.gamma);

    return config;
}

void print_config(const Config& config) {
    constexpr int label_width = 18;

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║          Langevin Dynamics Config          ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";

    std::cout << "[grid]\n";
    std::cout << "  " << std::left << std::setw(label_width) << "N"
              << " = " << config.grid.N << "\n\n";

    std::cout << "[conventions]\n";
    std::cout << "  " << std::left << std::setw(label_width) << "m"
              << " = " << config.conventions.m << "\n";
    std::cout << "  " << std::left << std::setw(label_width) << "kB"
              << " = " << config.conventions.kB << "\n\n";

    std::cout << "[time]\n";
    std::cout << "  " << std::left << std::setw(label_width) << "N"
              << " = " << config.time.N << "\n";
    std::cout << "  " << std::left << std::setw(label_width) << "end_time"
              << " = " << config.time.end_time << "\n";
    std::cout << "  " << std::left << std::setw(label_width) << "save_every"
              << " = " << config.time.save_every << "\n\n";

    std::cout << "[ensemble]\n";
    std::cout << "  " << std::left << std::setw(label_width) << "N"
              << " = " << config.ensemble.N << "\n\n";

    std::cout << "[model]\n";
    std::cout << "  " << std::left << std::setw(label_width) << "omega"
              << " = " << config.model.omega << "\n";
    std::cout << "  " << std::left << std::setw(label_width) << "beta"
              << " = " << config.model.beta << "\n";
    std::cout << "  " << std::left << std::setw(label_width) << "EJ"
              << " = " << config.model.EJ << "\n";
    std::cout << "  " << std::left << std::setw(label_width) << "gamma"
              << " = " << config.model.gamma << "\n\n";

    std::cout << "──────────────────────────────────────────────\n";
    std::cout << "Simulation setup complete.\n";
    std::cout << "──────────────────────────────────────────────\n\n";
}