cmake_minimum_required(VERSION 3.13)

project(langevin_dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# source directory
set(SOURCE_DIR src)

# cpp files
add_executable(langevin_dynamics
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/input/input.cpp
)

target_include_directories(langevin_dynamics PRIVATE 
    ${SOURCE_DIR}
)