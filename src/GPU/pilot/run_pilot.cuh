#pragma once

#include "input/input.hpp"
#include "potentials.hpp"
#include "run_pilot_simulation.cuh"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

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

    double worst_error_ratio = 0.0;

    double worst_absolute_difference = 0.0;
    double worst_difference_standard_error = 0.0;
    double worst_uncertainty_bound = 0.0;
    double worst_allowed_difference = 0.0;

    double worst_coarse_mean = 0.0;
    double worst_fine_mean = 0.0;

    std::size_t worst_index = 0;
};

inline TimeDiscretizationReport estimate_time_error(const ProcessedData &coarse,
                                                    const ProcessedData &fine,
                                                    const double relative_tolerance,
                                                    const double absolute_tolerance,
                                                    const double z_score = 2.0) {
    if (coarse.mean.size() != fine.mean.size() ||
        coarse.standard_error.size() != coarse.mean.size() ||
        fine.standard_error.size() != fine.mean.size()) {

        throw std::invalid_argument("ProcessedData vectors have inconsistent sizes");
    }

    if (!(relative_tolerance >= 0.0) || !(absolute_tolerance >= 0.0) || !(z_score > 0.0)) {

        throw std::invalid_argument("Invalid time-discretization tolerances");
    }

    TimeDiscretizationReport report;

    for (std::size_t i = 0; i < coarse.mean.size(); ++i) {
        const double coarse_mean = coarse.mean[i];
        const double fine_mean = fine.mean[i];

        const double coarse_se = coarse.standard_error[i];
        const double fine_se = fine.standard_error[i];

        if (!std::isfinite(coarse_mean) || !std::isfinite(fine_mean) || !std::isfinite(coarse_se) ||
            !std::isfinite(fine_se) || coarse_se < 0.0 || fine_se < 0.0) {

            report.passed = false;
            report.worst_error_ratio = std::numeric_limits<double>::infinity();
            report.worst_index = i;
            return report;
        }

        const double absolute_difference = std::abs(coarse_mean - fine_mean);
        const double difference_standard_error = std::hypot(coarse_se, fine_se);

        // Conservative upper bound on the plausible timestep discrepancy.
        const double uncertainty_bound = absolute_difference + z_score * difference_standard_error;

        // Symmetric scale: avoids treating either solution as exact.
        const double solution_scale = std::max(std::abs(coarse_mean), std::abs(fine_mean));

        const double allowed_difference = absolute_tolerance + relative_tolerance * solution_scale;

        const double error_ratio = allowed_difference > 0.0
                                       ? uncertainty_bound / allowed_difference
                                       : std::numeric_limits<double>::infinity();

        if (error_ratio > report.worst_error_ratio) {
            report.worst_error_ratio = error_ratio;
            report.worst_absolute_difference = absolute_difference;
            report.worst_difference_standard_error = difference_standard_error;
            report.worst_uncertainty_bound = uncertainty_bound;
            report.worst_allowed_difference = allowed_difference;
            report.worst_coarse_mean = coarse_mean;
            report.worst_fine_mean = fine_mean;
            report.worst_index = i;
        }

        if (uncertainty_bound > allowed_difference) {
            report.passed = false;
        }
    }

    return report;
}

inline ProcessedData
process_simulation_data(const std::vector<double> &batches /* [statistical_batches * N] */,
                        const std::size_t statistical_batches, const std::size_t N) {

    // possibly use welfords algorithm

    if (statistical_batches < 2) {
        throw std::invalid_argument("At least two statistical batches are required");
    }
    // initialize per site results
    ProcessedData result{std::vector<double>(N, 0.0), std::vector<double>(N, 0.0),
                         std::vector<double>(N, 0.0)};

    const double batch_count = static_cast<double>(statistical_batches);

    // calculate mean per site
    for (std::size_t batch = 0; batch < statistical_batches; ++batch) {

        const std::size_t batch_offset = batch * N;
        for (std::size_t site = 0; site < N; ++site) {
            result.mean[site] += batches[batch_offset + site];
        }
    }
    for (double &value : result.mean) {
        value /= batch_count; // mean
    }

    // calculate variance per site
    for (std::size_t batch = 0; batch < statistical_batches; ++batch) {

        const std::size_t batch_offset = batch * N;
        for (std::size_t site = 0; site < N; ++site) {
            const double deviation = batches[batch_offset + site] - result.mean[site];
            result.standard_deviation[site] += deviation * deviation; // unnormalized variance
        }
    }

    // Bessel correction (we estimated mean from dataset)
    const double bessel_correction = static_cast<double>(statistical_batches - 1);

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

    double worst_error_ratio = 0.0;
    double worst_uncertainty = 0.0;
    double worst_allowed_uncertainty = 0.0;

    std::size_t worst_index = 0;
};

inline StochasticReport estimate_stochastic_error(const ProcessedData &data,
                                                  const double relative_tolerance,
                                                  const double absolute_tolerance,
                                                  const double z_score = 2.0) {
    if (data.mean.size() != data.standard_error.size() ||
        data.standard_deviation.size() != data.mean.size()) {
        throw std::invalid_argument("ProcessedData vectors have inconsistent sizes");
    }

    if (!(relative_tolerance >= 0.0) || !(absolute_tolerance >= 0.0) || !(z_score > 0.0)) {
        throw std::invalid_argument("Invalid stochastic-error tolerances");
    }

    StochasticReport report;

    for (std::size_t i = 0; i < data.mean.size(); ++i) {
        const double mean = data.mean[i];
        const double se = data.standard_error[i];

        if (!std::isfinite(mean) || !std::isfinite(se) || se < 0.0) {
            report.passed = false;
            report.worst_error_ratio = std::numeric_limits<double>::infinity();
            report.worst_index = i;
            return report;
        }

        const double uncertainty = z_score * se;

        const double allowed_uncertainty = absolute_tolerance + relative_tolerance * std::abs(mean);

        const double error_ratio = allowed_uncertainty > 0.0
                                       ? uncertainty / allowed_uncertainty
                                       : std::numeric_limits<double>::infinity();

        if (error_ratio > report.worst_error_ratio) {
            report.worst_error_ratio = error_ratio;
            report.worst_uncertainty = uncertainty;
            report.worst_allowed_uncertainty = allowed_uncertainty;
            report.worst_index = i;
        }

        if (uncertainty > allowed_uncertainty) {
            report.passed = false;
        }
    }

    return report;
}

inline void print_stochastic_report(const char *name, const ProcessedData &data,
                                    const StochasticReport &report) {
    constexpr int indent_width = 4;
    constexpr int label_width = 15;

    const std::string indent(indent_width, ' ');
    const std::size_t i = report.worst_index;

    std::cout << '\n'
              << indent << "---------------\n"
              << indent << name << '\n'
              << indent << std::left << std::setw(label_width) << "Passed:" << std::boolalpha
              << report.passed << '\n'
              << indent << std::setw(label_width) << "Worst index:" << i << '\n'
              << indent << std::setw(label_width) << "Mean:" << data.mean[i] << '\n'
              << indent << std::setw(label_width) << "Sigma batch:" << data.standard_deviation[i]
              << '\n'
              << indent << std::setw(label_width) << "SE:" << data.standard_error[i] << '\n'
              << indent << std::setw(label_width) << "Uncertainty:" << report.worst_uncertainty
              << '\n'
              << indent << std::setw(label_width)
              << "Allowed uncertainty:" << report.worst_allowed_uncertainty << '\n'
              << indent << std::setw(label_width) << "Error ratio:" << report.worst_error_ratio
              << '\n';
}

inline void print_time_report(const char *name, const TimeDiscretizationReport &report) {
    constexpr int indent_width = 4;
    constexpr int label_width = 28;

    const std::string indent(indent_width, ' ');

    std::cout << '\n'
              << indent << "----------------\n"
              << indent << name << '\n'
              << indent << std::left << std::setw(label_width) << "Passed:" << std::boolalpha
              << report.passed << '\n'
              << indent << std::setw(label_width) << "Worst index:" << report.worst_index << '\n'
              << indent << std::setw(label_width) << "Coarse mean:" << report.worst_coarse_mean
              << '\n'
              << indent << std::setw(label_width) << "Fine mean:" << report.worst_fine_mean << '\n'
              << indent << std::setw(label_width)
              << "Absolute difference:" << report.worst_absolute_difference << '\n'
              << indent << std::setw(label_width)
              << "Difference SE:" << report.worst_difference_standard_error << '\n'
              << indent << std::setw(label_width)
              << "Uncertainty bound:" << report.worst_uncertainty_bound << '\n'
              << indent << std::setw(label_width)
              << "Allowed difference:" << report.worst_allowed_difference << '\n'
              << indent << std::setw(label_width) << "Error ratio:" << report.worst_error_ratio
              << '\n';
}

template <class Potential> inline PilotOutcome run_pilot(const Config &config) {

    const Potential potential(config);

    PilotOutcome pilot_outcome;

    // extract input params
    const int N = config.grid.N;

    const double m = config.conventions.m;

    const double end_time = config.time.end_time;
    const int N_time = config.time.N;
    const double dt = end_time / static_cast<double>(N_time);

    const int N_ensemble = config.ensemble.N;

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
        std::cout << "1. Checking generic FPU stability limit...\n";

        double r_max = 10; // TODO: add good estimation -> look at result Potential Energy and
                           // extract biggest r -> update r_max after new dt of course
        const double stability_limit = potential.stability_limit(r_max, m);

        // while
        constexpr double security_factor = 0.1;
        if (!(dt < security_factor * stability_limit)) {
            const int N_time_min = end_time / (security_factor * stability_limit) + 1;
            const double max_dt_stable = stability_limit * security_factor;
            std::ostringstream message;
            message << "Failed !\nChosen dt = " << dt
                    << " is too large. The maximum allowed timestep "
                    << "with safety factor of " << security_factor << " is " << max_dt_stable
                    << ". "
                    << "The formal stability limit is " << stability_limit << ". "
                    << "Choose N_time >= " << N_time_min << ".\n"
                    << "See docs/stability_analysis.md for more info.\n";
            return {Flag::FailedStabilityLimit, 0.0, message.str()};
        }
        std::cout << "Success!\n";
        std::cout << "Selected dt = " << dt << "\n";

        // configure reasonable duration (change N_time and end_time)
        const int target_steps = 10'000; // N_time of pilot run
        const double end_time_pilot = static_cast<double>(target_steps) * dt;
        std::cout << "\n\nTo ensure a limited running time for the pilot analysis, the end_time "
                     "for the pilot run is manually set to "
                  << end_time_pilot << "\n";

        // run dedicated GPU verison that returns observables * [statistic_batches * N]
        // all integrate to SAME end_time_pilot
        const int statistical_batches = 100;
        if ((N_ensemble % statistical_batches) != 0) {
            std::ostringstream message;
            message << "N_ensemble must be divisible by " << statistical_batches << "!\n";
            throw std::logic_error(message.str());
        }
        const int trajectories_per_statistical_batch =
            static_cast<int>(N_ensemble / statistical_batches);
        SimulationResults pilot_h; // contains observables * [statistical_batches * N] doubles
        SimulationResults pilot_h2;
        SimulationResults pilot_h4;
        std::cout << "Running the pilot simulations...\n\n";
        std::cout << "Running the first simulation with dt = " << dt << "\n";
        pilot_h = run_pilot_simulation<Potential>(statistical_batches, dt, target_steps, config,
                                                  trajectories_per_statistical_batch);
        std::cout << "Finished the first simulation.\n\n";
        std::cout << "Running the simulation with dt = " << dt / 2.0 << "\n";
        pilot_h2 = run_pilot_simulation<Potential>(statistical_batches, dt / 2.0, target_steps * 2,
                                                   config, trajectories_per_statistical_batch);
        std::cout << "Finished the second simulation.\n\n";
        std::cout << "Running the simulation with dt = " << dt / 4.0 << "\n";
        pilot_h4 = run_pilot_simulation<Potential>(statistical_batches, dt / 4.0, target_steps * 4,
                                                   config, trajectories_per_statistical_batch);
        std::cout << "Finished the third simulation.\n\n";
        std::cout << "Pilot simulations finished.\n\n";

        // reduce data from [batches * N] to [N]
        const ProcessedData total_h =
            process_simulation_data(pilot_h.total_energy, statistical_batches, N);
        const ProcessedData kinetic_h =
            process_simulation_data(pilot_h.kinetic_energy, statistical_batches, N);
        const ProcessedData potential_h =
            process_simulation_data(pilot_h.potential_energy, statistical_batches, N);
        const ProcessedData mean_h =
            process_simulation_data(pilot_h.tot_energy_mean, statistical_batches, 1);
        const ProcessedData spread_h =
            process_simulation_data(pilot_h.tot_energy_spread, statistical_batches, 1);

        const ProcessedData total_h2 =
            process_simulation_data(pilot_h2.total_energy, statistical_batches, N);
        const ProcessedData kinetic_h2 =
            process_simulation_data(pilot_h2.kinetic_energy, statistical_batches, N);
        const ProcessedData potential_h2 =
            process_simulation_data(pilot_h2.potential_energy, statistical_batches, N);
        const ProcessedData mean_h2 =
            process_simulation_data(pilot_h2.tot_energy_mean, statistical_batches, 1);
        const ProcessedData spread_h2 =
            process_simulation_data(pilot_h2.tot_energy_spread, statistical_batches, 1);

        const ProcessedData total_h4 =
            process_simulation_data(pilot_h4.total_energy, statistical_batches, N);
        const ProcessedData kinetic_h4 =
            process_simulation_data(pilot_h4.kinetic_energy, statistical_batches, N);
        const ProcessedData potential_h4 =
            process_simulation_data(pilot_h4.potential_energy, statistical_batches, N);
        const ProcessedData mean_h4 =
            process_simulation_data(pilot_h4.tot_energy_mean, statistical_batches, 1);
        const ProcessedData spread_h4 =
            process_simulation_data(pilot_h4.tot_energy_spread, statistical_batches, 1);

        // -----------------------------------------------------------------------
        // --------------------------
        // |    STOCHASTIC ERROR    |
        // --------------------------
        std::cout << "\n2. Checking stochastic error...";
        // estimate worst stochastic error
        constexpr double stochastic_relative_tolerance = 0.03;
        constexpr double stochastic_absolute_tolerance = 1.0e-2;

        constexpr double time_relative_tolerance = 0.05;
        constexpr double time_absolute_tolerance = 1.0e-2;

        const StochasticReport total_stoachstic_report_h = estimate_stochastic_error(
            total_h, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport kinetic_stoachstic_report_h = estimate_stochastic_error(
            kinetic_h, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport potential_stoachstic_report_h = estimate_stochastic_error(
            potential_h, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport mean_stoachstic_report_h = estimate_stochastic_error(
            mean_h, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport spread_stoachstic_report_h = estimate_stochastic_error(
            spread_h, stochastic_relative_tolerance, stochastic_absolute_tolerance);

        const StochasticReport total_stochastic_report_h2 = estimate_stochastic_error(
            total_h2, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport kinetic_stochastic_report_h2 = estimate_stochastic_error(
            kinetic_h2, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport potential_stochastic_report_h2 = estimate_stochastic_error(
            potential_h2, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport mean_stochastic_report_h2 = estimate_stochastic_error(
            mean_h2, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport spread_stochastic_report_h2 = estimate_stochastic_error(
            spread_h2, stochastic_relative_tolerance, stochastic_absolute_tolerance);

        const StochasticReport total_stochastic_report_h4 = estimate_stochastic_error(
            total_h4, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport kinetic_stochastic_report_h4 = estimate_stochastic_error(
            kinetic_h4, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport potential_stochastic_report_h4 = estimate_stochastic_error(
            potential_h4, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport mean_stochastic_report_h4 = estimate_stochastic_error(
            mean_h4, stochastic_relative_tolerance, stochastic_absolute_tolerance);
        const StochasticReport spread_stochastic_report_h4 = estimate_stochastic_error(
            spread_h4, stochastic_relative_tolerance, stochastic_absolute_tolerance);

        // check reports
        const bool stochastic_error_small_enough_h =
            total_stoachstic_report_h.passed && kinetic_stoachstic_report_h.passed &&
            potential_stoachstic_report_h.passed && mean_stoachstic_report_h.passed &&
            spread_stoachstic_report_h.passed;
        const bool stochastic_error_small_enough_h2 =
            total_stochastic_report_h2.passed && kinetic_stochastic_report_h2.passed &&
            potential_stochastic_report_h2.passed && mean_stochastic_report_h2.passed &&
            spread_stochastic_report_h2.passed;
        const bool stochastic_error_small_enough_h4 =
            total_stochastic_report_h4.passed && kinetic_stochastic_report_h4.passed &&
            potential_stochastic_report_h4.passed && mean_stochastic_report_h4.passed &&
            spread_stochastic_report_h4.passed;

        // stochastic reports
        std::cout << "\n\n=======Stochastic Report=======\n";
        print_stochastic_report("total energy h", total_h, total_stoachstic_report_h);
        print_stochastic_report("mean energy h", mean_h, mean_stoachstic_report_h);
        print_stochastic_report("spread of energy h", spread_h, spread_stoachstic_report_h);

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
        std::cout << "\n\n\n3. Checking time error...";

        const TimeDiscretizationReport total_time_h2_h4 = estimate_time_error(
            total_h2, total_h4, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport kinetic_time_h2_h4 = estimate_time_error(
            kinetic_h2, kinetic_h4, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport potential_time_h2_h4 = estimate_time_error(
            potential_h2, potential_h4, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport mean_time_h2_h4 =
            estimate_time_error(mean_h2, mean_h4, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport spread_time_h2_h4 = estimate_time_error(
            spread_h2, spread_h4, time_relative_tolerance, time_absolute_tolerance);

        // check whether h2 agrees with h4
        const bool h2_h4_agree = total_time_h2_h4.passed && kinetic_time_h2_h4.passed &&
                                 potential_time_h2_h4.passed && mean_time_h2_h4.passed &&
                                 spread_time_h2_h4.passed;

        // print h2_h4 report
        std::cout << "\n\n========Time Report========\n";
        print_time_report("total energy h2_h4", total_time_h2_h4);
        print_time_report("kinetic energy h2_h4", kinetic_time_h2_h4);
        print_time_report("potential energy h2_h4", potential_time_h2_h4);
        print_time_report("mean energy h2_h4", mean_time_h2_h4);
        print_time_report("spread of energy h2_h4", spread_time_h2_h4);

        // if this already fails, then h2 does not resolve fine enough
        if (!h2_h4_agree) {
            return {Flag::IncreaseN_time, dt / 2.0,
                    "Time error too big! Already h2 doesn't resolve the system enough. Try N_time "
                    "= {end_time / dt / 4.0 }"};
        }

        // check if h agrees with h2
        const TimeDiscretizationReport total_time_h_h2 = estimate_time_error(
            total_h, total_h2, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport kinetic_time_h_h2 = estimate_time_error(
            kinetic_h, kinetic_h2, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport potential_time_h_h2 = estimate_time_error(
            potential_h, potential_h2, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport mean_time_h_h2 =
            estimate_time_error(mean_h, mean_h2, time_relative_tolerance, time_absolute_tolerance);
        const TimeDiscretizationReport spread_time_h_h2 = estimate_time_error(
            spread_h, spread_h2, time_relative_tolerance, time_absolute_tolerance);

        const bool h_h2_agree = total_time_h_h2.passed && kinetic_time_h_h2.passed &&
                                potential_time_h_h2.passed && mean_time_h_h2.passed &&
                                spread_time_h_h2.passed;
        // if h does not agrees with h2
        if (!h_h2_agree) {
            std::cout
                << "\n\nh doesn't resolve the simulation, but h2 agreed with h4. Calculating h8 "
                   "to verify...\n";
            SimulationResults pilot_h8;
            std::cout << "Running the simulation with dt = " << dt / 8.0 << "\n";
            pilot_h8 =
                run_pilot_simulation<Potential>(statistical_batches, dt / 8.0, target_steps * 8,
                                                config, trajectories_per_statistical_batch);
            std::cout << "Finished the pilot h8 simulation.\n\n";

            const ProcessedData total_h8 =
                process_simulation_data(pilot_h8.total_energy, statistical_batches, N);
            const ProcessedData kinetic_h8 =
                process_simulation_data(pilot_h8.kinetic_energy, statistical_batches, N);
            const ProcessedData potential_h8 =
                process_simulation_data(pilot_h8.potential_energy, statistical_batches, N);
            const ProcessedData mean_h8 =
                process_simulation_data(pilot_h8.tot_energy_mean, statistical_batches, 1);
            const ProcessedData spread_h8 =
                process_simulation_data(pilot_h8.tot_energy_spread, statistical_batches, 1);

            const TimeDiscretizationReport total_time_h4_h8 = estimate_time_error(
                total_h4, total_h8, time_relative_tolerance, time_absolute_tolerance);
            const TimeDiscretizationReport kinetic_time_h4_h8 = estimate_time_error(
                kinetic_h4, kinetic_h8, time_relative_tolerance, time_absolute_tolerance);
            const TimeDiscretizationReport potential_time_h4_h8 = estimate_time_error(
                potential_h4, potential_h8, time_relative_tolerance, time_absolute_tolerance);
            const TimeDiscretizationReport mean_time_h4_h8 = estimate_time_error(
                mean_h4, mean_h8, time_relative_tolerance, time_absolute_tolerance);
            const TimeDiscretizationReport spread_time_h4_h8 = estimate_time_error(
                spread_h4, spread_h8, time_relative_tolerance, time_absolute_tolerance);

            const bool h4_h8_agree = total_time_h4_h8.passed && kinetic_time_h4_h8.passed &&
                                     potential_time_h4_h8.passed && mean_time_h4_h8.passed &&
                                     spread_time_h4_h8.passed;
            if (h4_h8_agree) {
                std::cout << "h4 and h8 agree. Therefore h2 should already be sufficient to "
                             "resolve the simulation.";
                if (h4_h8_agree) {
                    return {Flag::IncreaseN_time, dt / 2.0,
                            "h4 and h8 agree. Therefore h2 should already be sufficient to "
                            "resolve the simulation. Use N_time >= " +
                                std::to_string(2 * N_time) + "."};
                }
            } else {
                return {Flag::IncreaseN_time, 0.0,
                        "Increse N_time to at least N_time = {end_time / (dt / 8.0)}"};
            }
        }

        // h2 is right now valid
        return {Flag::AcceptedRequestedConfig, dt, "Pilot validated your config!"};
        // check if convergence of order two is visible (must be valid here!)
        // TODO: compare stochastic error to time error:
        // time error must be much bigger then stochastic error

    } else if constexpr (std::is_same_v<std::remove_cvref_t<Potential>, JosephsonPotential>) {
        // Josephson
        // repeat same logic

    } else {
        static_assert(std::is_same_v<Potential, void>, "Unsupported potential type");
    }
}
