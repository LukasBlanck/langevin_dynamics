#include "../potentials.hpp"
#include "GPU/pilot/run_pilot.cuh"
#include "run_simulation.cuh"
#include <iostream>
#include <stdexcept>

int main() {

    try {
        // load input params
        const Config config = load_config("src/input/input.toml");
        print_config(config);

        // create output_path for results
        const std::string output_path = "results/raw/GPU/local_energy.nc";

        // choose potential and run simulation
        if (config.model.potential == "FPU") {
            const FPUPotential potential(config);

            // run pilot
            const PilotOutcome pilot_outcome = run_pilot<FPUPotential>(config, potential);
            std::cout << pilot_outcome.message << "\n" << std::flush;

            // depending on pilot outcome:
            switch (pilot_outcome.decision) {
            case Flag::AcceptedRequestedConfig:
                run_simulation<FPUPotential>(config, output_path);
                break;

            case Flag::FailedStabilityLimit:
                return 1;
            case Flag::IncreaseN_ensemble:
                return 1;
            case Flag::IncreaseN_time:
                return 1;
            case Flag::SomethingWentWrong:
                throw std::runtime_error("Pilot simulation failed unexpectedly.");
            }

            // Josephson
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
