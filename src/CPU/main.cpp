#include "../potentials.hpp"
#include "run_simulation.hpp"
#include <stdexcept>

int main() {

    try {
        // load input params
        const Config config = load_config("src/input/input.toml");
        print_config(config);

        // create output_path for results
        const std::string output_path = "results/raw/CPU/local_energy.nc";

        // choose potential and run simulation
        if (config.model.potential == "FPU") {
            run_simulation<FPUPotential>(config, output_path);
        } else if (config.model.potential == "Josephson") {
            run_simulation<JosephsonPotential>(config, output_path);
        } else {
            throw std::runtime_error(
                "No valid potential chosen. Valid potentials are: FPU, Josephson");
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}