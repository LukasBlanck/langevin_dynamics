// this file will run before the demanded production run:
// if FPU requested:
// run a dedicated GPU/CPU implementation that monitors r_max with
// demanded dt and returns observables
// check if dt <= stability bound
// if yes:
// 1. run dedicated GPU version with dt/2 and dt/4 that returns observables
// check if observables stochastically match: result array [N] should match with 1%? to other dt/2..
// results: with batch_mean and overall mean and sigma if stochastic error smaller then threshold:
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
#include "potentials.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

struct PilotResults {
    std::vector<double> total_energy_batches;
    std::vector<double> kinetic_energy_batches;
    std::vector<double> potential_energy_batches;
    std::vector<double> energy_spread_batches;
    std::vector<double> energy_mean_batches;
};

enum class PilotDecision { AcceptedRequestedDt, ReduceDt, IncreaseEnsemble, ModifiedDt, SomethingWentWrong };

struct PilotOutcome {
    PilotDecision decision;
    double selected_dt;
};

struct PilotData {
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

inline TimeDiscretizationReport check_time_discretization(const PilotData &coarse,
                                                          const PilotData &fine,
                                                          const double relative_tolerance,
                                                          const double absolute_floor,
                                                          const double z_score = 2.0) {
    if (coarse.mean.size() != fine.mean.size() ||
        coarse.standard_error.size() != coarse.mean.size() ||
        fine.standard_error.size() != fine.mean.size()) {
        throw std::invalid_argument("PilotData vectors have inconsistent sizes");
    }
    if (!(relative_tolerance > 0.0)) {
        throw std::invalid_argument("relative_tolerance must be positive");
    }
    if (!(absolute_floor > 0.0)) {
        throw std::invalid_argument("absolute_floor must be positive");
    }

    TimeDiscretizationReport report;

    for (std::size_t i = 0; i < coarse.mean.size(); ++i) {
        const double absolute_difference = std::abs(coarse.mean[i] - fine.mean[i]);

        // Appropriate when the h and h/2 pilot runs are statistically
        // independent, which they are.
        const double difference_standard_error =
            std::hypot(coarse.standard_error[i], fine.standard_error[i]);

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

inline PilotData
estimate_from_batches(const std::vector<double> &batches /* [number_of_batches * N] */,
                      const std::size_t number_of_batches, const std::size_t N) {

    // possibly use welfords algorithm

    if (number_of_batches < 2) {
        throw std::invalid_argument("At least two statistical batches are required");
    }
    // initialize per site results
    PilotData result{std::vector<double>(N, 0.0), std::vector<double>(N, 0.0),
                     std::vector<double>(N, 0.0)};

    const double batch_count = static_cast<double>(number_of_batches);

    // Mean over statistical batches, independently for each site.
    for (std::size_t batch = 0; batch < number_of_batches; ++batch) {

        const std::size_t batch_offset = batch * N;
        for (std::size_t site = 0; site < N; ++site) {
            result.mean[site] += batches[batch_offset + site];
        }
    }
    for (double &value : result.mean) {
        value /= batch_count;
    }
    // Sample variance over batches, independently for each site.
    for (std::size_t batch = 0; batch < number_of_batches; ++batch) {

        const std::size_t batch_offset = batch * N;
        for (std::size_t site = 0; site < N; ++site) {
            const double deviation = batches[batch_offset + site] - result.mean[site];
            result.standard_deviation[site] += deviation * deviation;
        }
    }

    const double variance_denominator = static_cast<double>(number_of_batches - 1);

    for (std::size_t site = 0; site < N; ++site) {
        const double variance = result.standard_deviation[site] / variance_denominator;
        const double standard_deviation = std::sqrt(variance);

        result.standard_deviation[site] = standard_deviation;
        result.standard_error[site] = standard_deviation / std::sqrt(batch_count);
    }
    return result;
}

struct StochasticReport {
    bool passed = true;
    double worst_relative_error = 0.0;
    std::size_t worst_index = 0;
};

inline StochasticReport check_sampling_precision(const PilotData &data,
                                                 const double relative_tolerance,
                                                 const double absolute_floor,
                                                 const double z_score = 2.0) {
    if (data.mean.size() != data.standard_error.size()) {
        throw std::invalid_argument("PilotData vectors have inconsistent sizes");
    }

    StochasticReport report;

    for (std::size_t i = 0; i < data.mean.size(); ++i) {

        const double scale = std::max(absolute_floor, std::abs(data.mean[i]));
        const double relative_error = z_score * data.standard_error[i] / scale;

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
        std::cout << "--- Checking generic FPU stability limit...";
        double r_max = 10; // TODO: add good estimation -> look at result Potential Energy and
                           // extract biggest r -> update r_max after new dt of course
        double dt_production = dt;
        const double stability_limit = potential.stability_limit(r_max, m);

        // while
        while (!(dt_production < stability_limit)) {
            std::cout << "Selected timestep is too large: dt = " << dt_production << '\n';
            dt_production *= 0.5;
        }

        std::cout << "The selected time_step that fits the stability condition for BAOAB: dt = "
                  << dt_production;
        std::cout << "This time_step corresponds to N_time = " << end_time / dt_production;

        // configure reasonable duration (change N_time and end_time)
        const int target_steps = 10000; // N_time of pilot run
        const double end_time_pilot = static_cast<double>(target_steps) * dt_production;
        std::cout << "To ensure a limited time for running the pilot, the end_time_pilot is "
                     "manually set to "
                  << end_time_pilot;

        // run dedicated GPU verison that returns observables * [statistic_batches * N]
        // all integrate to SAME end_time_pilot
        const int number_of_batches = 100;
        PilotResults pilot_h; // contains observables * [number_of_batches * N] doubles
        PilotResults pilot_h2;
        PilotResults pilot_h4;
        std::cout << "Running the simulation with dt = " << dt_production;
        pilot_h = run_simulation_for_pilot(number_of_batches, dt_production, target_steps);
        std::cout << "Running the simulation with dt = " << dt_production / 2.0;
        pilot_h2 =
            run_simulation_for_pilot(number_of_batches, dt_production / 2.0, target_steps * 2);
        std::cout << "Running the simulation with dt = " << dt_production / 4.0;
        pilot_h4 =
            run_simulation_for_pilot(number_of_batches, dt_production / 4.0, target_steps * 4);
        std::cout << "Pilot simulations finished.";

        // check stochastic error
        const PilotData total_h =
            estimate_from_batches(pilot_h.total_energy_batches, number_of_batches, N);
        const PilotData kinetic_h =
            estimate_from_batches(pilot_h.kinetic_energy_batches, number_of_batches, N);
        const PilotData potential_h =
            estimate_from_batches(pilot_h.potential_energy_batches, number_of_batches, N);
        const PilotData mean_h =
            estimate_from_batches(pilot_h.energy_mean_batches, number_of_batches, 1);
        const PilotData spread_h =
            estimate_from_batches(pilot_h.energy_spread_batches, number_of_batches, 1);

        const PilotData total_h2 =
            estimate_from_batches(pilot_h2.total_energy_batches, number_of_batches, N);
        const PilotData kinetic_h2 =
            estimate_from_batches(pilot_h2.kinetic_energy_batches, number_of_batches, N);
        const PilotData potential_h2 =
            estimate_from_batches(pilot_h2.potential_energy_batches, number_of_batches, N);
        const PilotData mean_h2 =
            estimate_from_batches(pilot_h2.energy_mean_batches, number_of_batches, 1);
        const PilotData spread_h2 =
            estimate_from_batches(pilot_h2.energy_spread_batches, number_of_batches, 1);

        const PilotData total_h4 =
            estimate_from_batches(pilot_h4.total_energy_batches, number_of_batches, N);
        const PilotData kinetic_h4 =
            estimate_from_batches(pilot_h4.kinetic_energy_batches, number_of_batches, N);
        const PilotData potential_h4 =
            estimate_from_batches(pilot_h4.potential_energy_batches, number_of_batches, N);
        const PilotData mean_h4 =
            estimate_from_batches(pilot_h4.energy_mean_batches, number_of_batches, 1);
        const PilotData spread_h4 =
            estimate_from_batches(pilot_h4.energy_spread_batches, number_of_batches, 1);

        // process
        // estimate worst stochastic error
        constexpr double sampling_tolerance = 0.005;
        constexpr double absolute_floor = 1.0e-12;

        const StochasticReport total_precision_h =
            check_sampling_precision(total_h, sampling_tolerance, absolute_floor);
        const StochasticReport kinetic_precision_h =
            check_sampling_precision(kinetic_h, sampling_tolerance, absolute_floor);
        const StochasticReport potential_precision_h =
            check_sampling_precision(potential_h, sampling_tolerance, absolute_floor);
        const StochasticReport mean_precision_h =
            check_sampling_precision(mean_h, sampling_tolerance, absolute_floor);
        const StochasticReport spread_precision_h =
            check_sampling_precision(spread_h, sampling_tolerance, absolute_floor);

        const StochasticReport total_precision_h2 =
            check_sampling_precision(total_h2, sampling_tolerance, absolute_floor);
        const StochasticReport kinetic_precision_h2 =
            check_sampling_precision(kinetic_h2, sampling_tolerance, absolute_floor);
        const StochasticReport potential_precision_h2 =
            check_sampling_precision(potential_h2, sampling_tolerance, absolute_floor);
        const StochasticReport mean_precision_h2 =
            check_sampling_precision(mean_h2, sampling_tolerance, absolute_floor);
        const StochasticReport spread_precision_h2 =
            check_sampling_precision(spread_h2, sampling_tolerance, absolute_floor);

        const StochasticReport total_precision_h4 =
            check_sampling_precision(total_h4, sampling_tolerance, absolute_floor);
        const StochasticReport kinetic_precision_h4 =
            check_sampling_precision(kinetic_h4, sampling_tolerance, absolute_floor);
        const StochasticReport potential_precision_h4 =
            check_sampling_precision(potential_h4, sampling_tolerance, absolute_floor);
        const StochasticReport mean_precision_h4 =
            check_sampling_precision(mean_h4, sampling_tolerance, absolute_floor);
        const StochasticReport spread_precision_h4 =
            check_sampling_precision(spread_h4, sampling_tolerance, absolute_floor);

        // TODO:  if every stochasticReport.passed == true, then we know that N_ensemble (for now)
        // is big enough, else throw runtim_error("Increase N_ensemble")

        // estimate time error
        constexpr double time_tolerance = 0.01; // 1%

        const TimeDiscretizationReport total_time_h2_h4 =
            check_time_discretization(total_h2, total_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport kinetic_time_h2_h4 =
            check_time_discretization(kinetic_h2, kinetic_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport potential_time_h2_h4 =
            check_time_discretization(potential_h2, potential_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport mean_time_h2_h4 =
            check_time_discretization(mean_h2, mean_h4, time_tolerance, absolute_floor);
        const TimeDiscretizationReport spread_time_h2_h4 =
            check_time_discretization(spread_h2, spread_h4, time_tolerance, absolute_floor);

        // check whether h2 agrees with h4
        const bool h2_h4_agree = total_time_h2_h4.passed && kinetic_time_h2_h4.passed &&
                                 potential_time_h2_h4.passed && mean_time_h2_h4.passed &&
                                 spread_time_h2_h4.passed;
        // if this already fails, then h2 does not resolve fine enough
        if (!h2_h4_agree) {
            return {PilotDecision::ReduceDt, dt_production / 2.0};
        }

        // check if h agrees with h2
        const TimeDiscretizationReport total_time_h_h2 =
            check_time_discretization(total_h, total_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport kinetic_time_h_h2 =
            check_time_discretization(kinetic_h, kinetic_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport potential_time_h_h2 =
            check_time_discretization(potential_h, potential_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport mean_time_h_h2 =
            check_time_discretization(mean_h, mean_h2, time_tolerance, absolute_floor);
        const TimeDiscretizationReport spread_time_h_h2 =
            check_time_discretization(spread_h, spread_h2, time_tolerance, absolute_floor);

        const bool h_h2_agree = total_time_h_h2.passed && kinetic_time_h_h2.passed &&
                                potential_time_h_h2.passed && mean_time_h_h2.passed &&
                                spread_time_h_h2.passed;
        // if h agrees with h2
        if (h_h2_agree) {
            return {PilotDecision::AcceptedRequestedDt, dt_production};
        }
        // should NEVER go till here
        return {PilotDecision::SomethingWentWrong, dt_production / 2.0};

        // check if convergence of order two is visible

    } else if constexpr (std::is_same_v<std::remove_cvref_t<Potential>, JosephsonPotential>) {
        // Josephson

    } else {
        static_assert(std::is_same_v<Potential, void>, "Unsupported potential type");
    }

    // // check time discretization error
    // error_abs_1 = pilot_observables_h - pilot_observables_h_half;
    // error_abs_2 = pilot_observables_h_half - pilot_observables_h_quarter;

    // if (time_error_small_enough) {
    //     check_convergence();
    //     if (second_order_onvergence) {
    //         std::cout << "You're good to go!";
    //         std::cout << "The production values are now: ";
    //         return true;
    //     } else {
    //         if (stochastic_error > time_error_small_enough)
    //             std::cout << "Stochastic error is bigger then time_error!\n Increase
    //             N_ensemble.";
    //     }
    //     else {
    //         std::cout << "Something went wrong...";
    //     }
    // } else {
    //     std::cout << "Time error is too big. Increase N_t";
    // }

    return PilotOutcome;
}
