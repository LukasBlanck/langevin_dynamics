#include <iostream>
#include "input/input.hpp"

// initialize grid -> q,p,F,T arrays m?
// create result arrays -> ej, pearson_corr

// ensemble

// evaluate forces
    // iterate over j
    // evaluate dV/dq
// update p with dt/2
// update q with dt/2

// update p with dt O.U step
// update q with dt/2
// evaluate forces
    // iterate over j
    // evaluate dV/dq
// update p with dt/2


// at extract times - write current state into result arrays

int main() {
    // load input params

    Config config = load_config("src/input/input.toml");
    std::cout << "Grid:" << config.grid.N;


    return 0;
}