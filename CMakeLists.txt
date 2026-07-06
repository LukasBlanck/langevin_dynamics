cmake_minimum_required(VERSION 3.13)

project(langevin_dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# source directory
set(SOURCE_DIR src)

# cpp files
add_executable(langevin_dynamics
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/input/input.cpp
)

# hpp files
target_include_directories(langevin_dynamics PRIVATE 
    ${SOURCE_DIR}
)

# netcdf
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF REQUIRED IMPORTED_TARGET netcdf)

target_include_directories(langevin_dynamics PRIVATE
    /opt/homebrew/include
    /opt/homebrew/opt/netcdf-cxx/include
)

target_link_libraries(langevin_dynamics PRIVATE
    PkgConfig::NETCDF
    /opt/homebrew/opt/netcdf-cxx/lib/libnetcdf-cxx4.dylib
)

# compilation warnings
target_compile_options(langevin_dynamics PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)