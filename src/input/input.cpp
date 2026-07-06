// config.cpp
#include "input.hpp"

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