#include "potentials.hpp"
#include "test_helpers.hpp"
#include <array>
#include <cstddef>
#include <gtest/gtest.h>

TEST(test_forces, fpu_and_josephson_force) {

    Config config = create_config();

    // create potentials
    FPUPotential fpu_potential(config);
    JosephsonPotential josephson_potential(config);

    const double epsilon = 1e-6;

    // create distance from local min
    std::array<double, 30> r;
    for (std::size_t i = 0; i < r.size(); i++) {
        r[i] = 0.1 * static_cast<double>(i + 1);
        // stencil force fpu
        double fpu_V_p_eps = fpu_potential.V(r[i] + epsilon);
        double fpu_V_m_eps = fpu_potential.V(r[i] - epsilon);

        double fpu_force_stencil = - (fpu_V_p_eps - fpu_V_m_eps) / (2 * epsilon);

        // stencil force josephson
        double jos_V_p_eps = josephson_potential.V(r[i] + epsilon);
        double jos_V_m_eps = josephson_potential.V(r[i] - epsilon);

        double jos_force_stencil = - (jos_V_p_eps - jos_V_m_eps) / (2 * epsilon);

        // force from langevin_library
        double lib_fpu_force = - fpu_potential.dV(r[i]);
        double lib_jos_force = - josephson_potential.dV(r[i]);

        EXPECT_NEAR(lib_fpu_force, fpu_force_stencil, 1e-8);
        EXPECT_NEAR(lib_jos_force, jos_force_stencil, 1e-8);
    }
}