#pragma once

#include "input/input.hpp"
#include "potentials.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

struct SimulationResults {
    std::vector<double> total_energy_batches;     // [batches * N]
    std::vector<double> kinetic_energy_batches;   // [batches * N]
    std::vector<double> potential_energy_batches; // [batches * N]
    std::vector<double> energy_spread_batches;    // [batches]
    std::vector<double> energy_mean_batches;      // [batches]
};

enum class Flag {
    FailedStabilityLimit,
    IncreaseN_ensemble,
    IncreaseN_time,
    AcceptedRequestedConfig,
    SomethingWentWrong
};

struct PilotOutcome {
    Flag decision;
    double selected_dt;
    std::string message;
};

struct ProcessedData {
    // vector of length N
    std::vector<double> mean;
    std::vector<double> standard_deviation; // sigma
    std::vector<double> standard_error;
};

struct TimeDiscretizationReport {
    bool passed = true;

    double worst_relative_difference = 0.0;
    double worst_absolute_difference = 0.0;
    double worst_difference_standard_error = 0.0;

    std::size_t worst_index = 0;
};

inline TimeDiscretizationReport estimate_time_error(const ProcessedData &coarse,
                                                    const ProcessedData &fine,
                                                    const double relative_tolerance,
                                                    const double absolute_floor,
                                                    const double z_score = 2.0) {
    if (coarse.mean.size() != fine.mean.size() ||
        coarse.standard_error.size() != coarse.mean.size() ||
        fine.standard_error.size() != fine.mean.size()) {
        throw std::invalid_argument("ProcessedData vectors have inconsistent sizes");
    }
    if (!(relative_tolerance > 0.0)) {
        throw std::invalid_argument("relative_tolerance must be positive");
    }
    if (!(absolute_floor > 0.0)) {
        throw std::invalid_argument("absolute_floor must be positive");
    }

    TimeDiscretizationReport report;

    for (std::size_t i = 0; i < coarse.mean.size() /* [N] */; ++i) {
        const double absolute_difference = std::abs(coarse.mean[i] - fine.mean[i]);

        // Appropriate when the h and h/2 pilot runs are statistically
        // independent, which they are.
        const double difference_standard_error =
            std::hypot(coarse.standard_error[i], fine.standard_error[i]); // sqrt(a^2 + b^2)

        const double scale = std::max(absolute_floor, std::abs(fine.mean[i]));
        // Conservative upper estimate of the plausible discrepancy.
        const double relative_difference =
            (absolute_difference + z_score * difference_standard_error) / scale;

        if (relative_difference > report.worst_relative_difference) {
            report.worst_relative_difference = relative_difference;
            report.worst_absolute_difference = absolute_difference;
            report.worst_difference_standard_error = difference_standard_error;
            report.worst_index = i;
        }
        if (relative_difference > relative_tolerance) {
            report.passed = false;
        }
    }
    return report;
}

inline ProcessedData
process_simulation_data(const std::vector<double> &batches /* [number_of_batches * N] */,
                        const std::size_t number_of_batches, const std::size_t N) {

    // possibly use welfords algorithm

    if (number_of_batches < 2) {
        throw std::invalid_argument("At least two statistical batches are required");
    }
    // initialize per site results
    ProcessedData result{std::vector<double>(N, 0.0), std::vector<double>(N, 0.0),
                         std::vector<double>(N, 0.0)};

    const double batch_count = static_cast<double>(number_of_batches);

    // calculate mean per site
    for (std::size_t batch = 0; batch < number_of_batches; ++batch) {

        const std::size_t batch_offset = batch * N;
        for (std::size_t site = 0; site < N; ++site) {
            result.mean[site] += batches[batch_offset + site];
        }
    }
    for (double &value : result.mean) {
        value /= batch_count; // mean
    }

    // calculate variance per site
    for (std::size_t batch = 0; batch < number_of_batches; ++batch) {

        const std::size_t batch_offset = batch * N;
        for (std::size_t site = 0; site < N; ++site) {
            const double deviation = batches[batch_offset + site] - result.mean[site];
            result.standard_deviation[site] += deviation * deviation; // unnormalized variance
        }
    }

    // Bessel correction (we estimated mean from dataset)
    const double bessel_correction = static_cast<double>(number_of_batches - 1);

    for (std::size_t site = 0; site < N; ++site) {
        const double variance = result.standard_deviation[site] / bessel_correction;
        const double standard_deviation = std::sqrt(variance);

        result.standard_deviation[site] = standard_deviation;
        result.standard_error[site] =
            standard_deviation / std::sqrt(batch_count); // uncertainty of mean
    }
    return result;
}

struct StochasticReport {
    bool passed = true;
    double worst_relative_error = 0.0;
    std::size_t worst_index = 0;
};

inline StochasticReport estimate_stochastic_error(const ProcessedData &processed_data /* [N]*/,
                                                  const double relative_tolerance,
                                                  const double absolute_floor,
                                                  const double z_score = 2.0) {
    if (processed_data.mean.size() != processed_data.standard_error.size()) {
        throw std::invalid_argument("ProcessedData vectors have inconsistent sizes");
    }

    StochasticReport report;

    for (std::size_t i = 0; i < processed_data.mean.size() /* [N] */; ++i) {

        const double scale = std::max(absolute_floor, std::abs(processed_data.mean[i]));
        const double relative_error =
            z_score * processed_data.standard_error[i] / scale; // small SE -> small relative_error

        if (relative_error > report.worst_relative_error) {
            report.worst_relative_error = relative_error;
            report.worst_index = i;
        }

        if (relative_error > relative_tolerance) {
            report.passed = false;
        }
    }
    return report;
}

template <class Potential>
inline PilotOutcome run_pilot(const Config &config, const Potential potential) {

    PilotOutcome pilot_outcome;

    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;
    const double kB = config.conventions.kB;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

    const double left_bath_T = config.model.left_bath_T;
    const double gamma = config.model.lambda / m;
    const double omega = config.model.omega;
    const double beta = config.model.beta;
    const double E_J = config.model.EJ;

    // check chosen potential
    if constexpr (std::is_same_v<std::remove_cvref_t<Potential>, FPUPotential>) {
        // FPU
        // check stability bond
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════╗\n";
        std::cout << "║                  Pilot Run                 ║\n";
        std::cout << "╚════════════════════════════════════════════╝\n";
        std::cout << "\n";
        std::cout << "1. Checking generic FPU stability limit...";

        double r_max = 10; // TODO: add good estimation -> look at result Potential Energy and
                           // extract biggest r -> update r_max after new dt of course
        const double stability_limit = potential.stability_limit(r_max, m);

        // while
        if (!(dt < stability_limit)) {
            return {Flag::FailedStabilityLimit, 0.0,
                    "Chosen dt = end_time / N_time is too large. It failed the formal stability "
                    "limit (find it under docs/stability_analysis.md). Chose N_time > {end_time / "
                    "stability_limit}"};
        }
        std::cout << "Success!\n";
        std::cout << "Selected dt = " << dt;

        // configure reasonable duration (change N_time and end_time)
        const int target_steps = 10'000; // N_time of pilot run
        const double end_time_pilot = static_cast<double>(target_steps) * dt;
        std::cout << "\n\nTo ensure a limited running time for the pilot analysis, the end_time "
                     "for the pilot run is manually set to "
                  << end_time_pilot << "\n";

        // run dedicated GPU verison that returns observables * [statistic_batches * N]
        // all integrate to SAME end_time_pilot
        const int number_of_batches = 100;
        SimulationResults pilot_h; // contains observables * [number_of_batches * N] doubles
        SimulationResults pilot_h2;
        SimulationResults pilot_h4;
        std::cout << "Running the pilot simulations...\n\n";
        std::cout << "Running the firt simulation with dt = " << dt / 2.0 << "\n";
        pilot_h = run_pilot_simulation(number_of_batches, dt, target_steps);
        std::cout << "Finished the first simulation.\n\n";
        std::cout << "Running the simulation with dt = " << dt / 2.0 << "\n";
        pilot_h2 = run_pilot_simulation(number_of_batches, dt / 2.0, target_steps * 2);
        std::cout << "Finished the second simulation.\n\n";
        std::cout << "Running the simulation with dt = " << dt / 4.0 << "\n";
        pilot_h4 = run_pilot_simulation(number_of_batches, dt / 4.0, target_steps * 4);
        std::cout << "Finished the third simulation.\n\n";
        std::cout << "Pilot simulations finished.";

        // reduce data from [batches * N] to [N]
        const ProcessedData total_h =
            process_simulation_data(pilot_h.total_energy_batches, number_of_batches, N);
        const ProcessedData kinetic_h =
            process_simulation_data(pilot_h.kinetic_energy_batches, number_of_batches, N);
        const ProcessedData potential_h =
            process_simulation_data(pilot_h.potential_energy_batches, number_of_batches, N);
        const ProcessedData mean_h =
            process_simulation_data(pilot_h.energy_mean_batches, number_of_batches, 1);
        const ProcessedData spread_h =
            process_simulation_data(pilot_h.energy_spread_batches, number_of_batches, 1);

        const ProcessedData total_h2 =
            process_simulation_data(pilot_h2.total_energy_batches, number_of_batches, N);
        const ProcessedData kinetic_h2 =
            process_simulation_data(pilot_h2.kinetic_energy_batches, number_of_batches, N);
        const ProcessedData potential_h2 =
            process_simulation_data(pilot_h2.potential_energy_batches, number_of_batches, N);
        const ProcessedData mean_h2 =
            process_simulation_data(pilot_h2.energy_mean_batches, number_of_batches, 1);
        const ProcessedData spread_h2 =
            process_simulation_data(pilot_h2.energy_spread_batches, number_of_batches, 1);

        const ProcessedData total_h4 =
            process_simulation_data(pilot_h4.total_energy_batches, number_of_batches, N);
        const ProcessedData kinetic_h4 =
            process_simulation_data(pilot_h4.kinetic_energy_batches, number_of_batches, N);
        const ProcessedData potential_h4 =
            process_simulation_data(pilot_h4.potential_energy_batches, number_of_batches, N);
        const ProcessedData mean_h4 =
            process_simulation_data(pilot_h4.energy_mean_batches, number_of_batches, 1);
        const ProcessedData spread_h4 =
            process_simulation_data(pilot_h4.energy_spread_batches, number_of_batches, 1);

        // -----------------------------------------------------------------------
        // --------------------------
        // |    STOCHASTIC ERROR    |
        // --------------------------
        std::cout << "2. Checking stochastic error...";
        // estimate worst stochastic error
        constexpr double relative_error_threshold = 0.05;
        constexpr double absolute_floor = 1.0e-5; // guard for avoiding division by zero / near zero

        const StochasticReport total_stoachstic_report_h =
            estimate_stochastic_error(total_h, relative_error_threshold, absolute_floor);
        const StochasticReport kinetic_stoachstic_report_h =
            estimate_stochastic_error(kinetic_h, relative_error_threshold, absolute_floor);
        const StochasticReport potential_stoachstic_report_h =
            estimate_stochastic_error(potential_h, relative_error_threshold, absolute_floor);
        const StochasticReport mean_stoachstic_report_h =
            estimate_stochastic_error(mean_h, relative_error_threshold, absolute_floor);
        const StochasticReport spread_stoachstic_report_h =
            estimate_stochastic_error(spread_h, relative_error_threshold, absolute_floor);

        const StochasticReport total_stochastic_report_h2 =
            estimate_stochastic_error(total_h2, relative_error_threshold, absolute_floor);
        const StochasticReport kinetic_stochastic_report_h2 =
            estimate_stochastic_error(kinetic_h2, relative_error_threshold, absolute_floor);
        const StochasticReport potential_stochastic_report_h2 =
            estimate_stochastic_error(potential_h2, relative_error_threshold, absolute_floor);
        const StochasticReport mean_stochastic_report_h2 =
            estimate_stochastic_error(mean_h2, relative_error_threshold, absolute_floor);
        const StochasticReport spread_stochastic_report_h2 =
            estimate_stochastic_error(spread_h2, relative_error_threshold, absolute_floor);

        const StochasticReport total_stochastic_report_h4 =
            estimate_stochastic_error(total_h4, relative_error_threshold, absolute_floor);
        const StochasticReport kinetic_stochastic_report_h4 =
            estimate_stochastic_error(kinetic_h4, relative_error_threshold, absolute_floor);
        const StochasticReport potential_stochastic_report_h4 =
            estimate_stochastic_error(potential_h4, relative_error_threshold, absolute_floor);
        const StochasticReport mean_stochastic_report_h4 =
            estimate_stochastic_error(mean_h4, relative_error_threshold, absolute_floor);
        const StochasticReport spread_stochastic_report_h4 =
            estimate_stochastic_error(spread_h4, relative_error_threshold, absolute_floor);

        // check reports
        const bool stochastic_error_small_enough_h =
            total_stoachstic_report_h.passed && kinetic_stoachstic_report_h.passed &&
            potential_stoachstic_report_h.passed && mean_stoachstic_report_h.passed &&
            spread_stoachstic_report_h.passed;
        const bool stochastic_error_small_enough_h2 =
            total_stochastic_report_h2.passed && kinetic_stochastic_report_h2.passed &&
            kinetic_stochastic_report_h2.passed && mean_stochastic_report_h2.passed &&
            spread_stochastic_report_h2.passed;
        const bool stochastic_error_small_enough_h4 =
            total_stochastic_report_h4.passed && kinetic_stochastic_report_h4.passed &&
            potential_stochastic_report_h4.passed && mean_stochastic_report_h4.passed &&
            spread_stochastic_report_h4.passed;

        if ((stochastic_error_small_enough_h && !stochastic_error_small_enough_h2) |
            (stochastic_error_small_enough_h && !stochastic_error_small_enough_h4)) {
            throw std::runtime_error("h has small enough stoachstic error. but h2 or h4 has bigger "
                                     "stochastic error -> inspect manually!");
        }
        if (!stochastic_error_small_enough_h) {
            return {Flag::IncreaseN_ensemble, 0.0,
                    "Stochastic error is too big! Increase N_ensemble and try again."};
        }

        // -----------------------------------------------------------------------
        // --------------------
        // |    TIME ERROR    |
        // --------------------
        std::cout << "3. Checking time error...";
        constexpr double time_tolerance = 0.005; // 0.5% = 0.1 * stochastic threshold

        const TimeDiscretizationReport total_time_h2_h4 =
            estimate_time_error(total_h2, total_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport kinetic_time_h2_h4 =
            estimate_time_error(kinetic_h2, kinetic_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport potential_time_h2_h4 =
            estimate_time_error(potential_h2, potential_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport mean_time_h2_h4 =
            estimate_time_error(mean_h2, mean_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport spread_time_h2_h4 =
            estimate_time_error(spread_h2, spread_h4, time_tolerance, absolute_floor);

        // check whether h2 agrees with h4
        const bool h2_h4_agree = total_time_h2_h4.passed && kinetic_time_h2_h4.passed &&
                                 potential_time_h2_h4.passed && mean_time_h2_h4.passed &&
                                 spread_time_h2_h4.passed;
        // if this already fails, then h2 does not resolve fine enough
        if (!h2_h4_agree) {
            return {Flag::IncreaseN_time, dt / 2.0,
                    "Time error too big! Already h2 doesn't resolve the system enough. Try N_time "
                    "= {end_time / dt / 4.0 }"};
        }

        // check if h agrees with h2
        const TimeDiscretizationReport total_time_h_h2 =
            estimate_time_error(total_h, total_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport kinetic_time_h_h2 =
            estimate_time_error(kinetic_h, kinetic_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport potential_time_h_h2 =
            estimate_time_error(potential_h, potential_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport mean_time_h_h2 =
            estimate_time_error(mean_h, mean_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport spread_time_h_h2 =
            estimate_time_error(spread_h, spread_h2, time_tolerance, absolute_floor);

        const bool h_h2_agree = total_time_h_h2.passed && kinetic_time_h_h2.passed &&
                                potential_time_h_h2.passed && mean_time_h_h2.passed &&
                                spread_time_h_h2.passed;
        // if h does not agrees with h2
        if (!h_h2_agree) {
            std::cout << "h doesn't resolve the simulation, but h2 agreed with h4. Calculating h8 "
                         "to verify...";
            SimulationResults pilot_h8;
            std::cout << "Running the simulation with dt = " << dt / 8.0 << "\n";
            pilot_h8 = run_pilot_simulation(number_of_batches, dt / 8.0, target_steps * 8);
            std::cout << "Finished the pilot simulation.\n\n";

            const ProcessedData total_h8 =
                process_simulation_data(pilot_h8.total_energy_batches, number_of_batches, N);
            const ProcessedData kinetic_h8 =
                process_simulation_data(pilot_h8.kinetic_energy_batches, number_of_batches, N);
            const ProcessedData potential_h8 =
                process_simulation_data(pilot_h8.potential_energy_batches, number_of_batches, N);
            const ProcessedData mean_h8 =
                process_simulation_data(pilot_h8.energy_mean_batches, number_of_batches, 1);
            const ProcessedData spread_h8 =
                process_simulation_data(pilot_h8.energy_spread_batches, number_of_batches, 1);

            const TimeDiscretizationReport total_time_h4_h8 =
                estimate_time_error(total_h4, total_h8, time_tolerance, absolute_floor);
            const TimeDiscretizationReport kinetic_time_h4_h8 =
                estimate_time_error(kinetic_h4, kinetic_h8, time_tolerance, absolute_floor);
            const TimeDiscretizationReport potential_time_h4_h8 =
                estimate_time_error(potential_h4, potential_h8, time_tolerance, absolute_floor);
            const TimeDiscretizationReport mean_time_h4_h8 =
                estimate_time_error(mean_h4, mean_h8, time_tolerance, absolute_floor);
            const TimeDiscretizationReport spread_time_h4_h8 =
                estimate_time_error(spread_h4, spread_h8, time_tolerance, absolute_floor);

            const bool h4_h8_agree = total_time_h4_h8.passed && kinetic_time_h4_h8.passed &&
                                     potential_time_h4_h8.passed && mean_time_h4_h8.passed &&
                                     spread_time_h4_h8.passed;
            if (h4_h8_agree) {
                std::cout << "h4 and h8 agree. Therefore h2 should already be sufficient to "
                             "resolve the simulation.";
            } else {
                return {Flag::IncreaseN_time, 0.0,
                        "Increse N_time to at least N_time = {end_time / (dt / 8.0)}"};
            }
        }

        // h2 is right now valid

        // check if convergence of order two is visible (must be valid here!)
        // TODO: compare stochastic error to time error:
        // time error must be much smaller then stochastic error

    } else if constexpr (std::is_same_v<std::remove_cvref_t<Potential>, JosephsonPotential>) {
        // Josephson
        // repeat same logic

    } else {
        static_assert(std::is_same_v<Potential, void>, "Unsupported potential type");
    }
}
