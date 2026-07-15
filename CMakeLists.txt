cmake_minimum_required(VERSION 3.20)

project(langevin_dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# source directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- begin netcdf
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF_C QUIET IMPORTED_TARGET netcdf)
pkg_check_modules(NETCDF_CXX QUIET IMPORTED_TARGET netcdf-cxx4)
# netcdf c fallback
if(NETCDF_C_FOUND)
    message(STATUS "Found NetCDF C via pkg-config")
    set(NETCDF_C_TARGET PkgConfig::NETCDF_C)
else()
    message(STATUS "NetCDF C not found via pkg-config; trying find_path/find_library")

    find_path(NETCDF_C_INCLUDE_DIR NAMES netcdf.h)
    find_library(NETCDF_C_LIBRARY NAMES netcdf)

    if(NOT NETCDF_C_INCLUDE_DIR OR NOT NETCDF_C_LIBRARY)
        message(FATAL_ERROR "Could not find NetCDF C library.\n"
            "Try loading a NetCDF module or setting CMAKE_PREFIX_PATH, e.g.\n"
            "  cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/netcdf"
        )
    else()
        message(STATUS "NetCDF C found in ${NETCDF_C_INCLUDE_DIR} or ${NETCDF_C_LIBRARY}")
    endif()

    # create netcdf c target
    add_library(NetCDF::C UNKNOWN IMPORTED)     # UNKNOWN: currently not important whether static or shared library was found
    set_target_properties(NetCDF::C PROPERTIES
        IMPORTED_LOCATION "${NETCDF_C_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_C_INCLUDE_DIR}"
    )
    set(NETCDF_C_TARGET NetCDF::C)
endif()

# netcdf cxx fallback
if(NETCDF_CXX_FOUND)
    message(STATUS "Found NetCDF C++ via pkg-config")
    set(NETCDF_CXX_TARGET PkgConfig::NETCDF_CXX)
else()
    message(STATUS "NetCDF C++ not found via pkg-config; trying find_path/find_library")

    find_path(NETCDF_CXX_INCLUDE_DIR NAMES netcdf)
    find_library(NETCDF_CXX_LIBRARY NAMES netcdf-cxx4 netcdf_c++4)

    if(NOT NETCDF_CXX_INCLUDE_DIR OR NOT NETCDF_CXX_LIBRARY)
        message(FATAL_ERROR "Could not find NetCDF C++4 library.\n"
            "Try loading a netcdf-cxx4 module or setting CMAKE_PREFIX_PATH, e.g.\n"
            "  cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/netcdf-cxx")
    else()
        message(STATUS "NetCDF C++ found in ${NETCDF_CXX_INCLUDE_DIR} or ${NETCDF_CXX_LIBRARY}")
    endif()

    # create netcdf cxx target
    add_library(NetCDF::CXX UNKNOWN IMPORTED)   # UNKNOWN: currently not important whether static or shared library was found
    set_target_properties(NetCDF::CXX PROPERTIES
        IMPORTED_LOCATION "${NETCDF_CXX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_CXX_INCLUDE_DIR}"
    )
    set(NETCDF_CXX_TARGET NetCDF::CXX)
endif()
# ---- end netcdf


# core library used by both: langevin_dynamics and langevin_tests
add_library(langevin_core STATIC        # core library
    ${SOURCE_DIR}/input/input.cpp
)

# hpp files (for core)
target_include_directories(langevin_core    
    PUBLIC
        ${SOURCE_DIR}
)

# linking netcdf to core
target_link_libraries(langevin_core
    PUBLIC
        ${NETCDF_CXX_TARGET}
        ${NETCDF_C_TARGET}
)

# compilation warnings
target_compile_options(langevin_core PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# ----------- create executables ----------
add_executable(langevin_dynamics
    ${SOURCE_DIR}/main.cpp)

target_link_libraries(langevin_dynamics
    PRIVATE
        langevin_core)

# test executable
option(BUILD_TESTING "Build tests" OFF)

if (BUILD_TESTING)
    include(CTest)

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
    if(TEST_SOURCES)
        include(FetchContent)

        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        
        FetchContent_MakeAvailable(googletest)

        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang emits a character-conversion warning inside GoogleTest's
            # own implementation, specifically around ImplicitCast_<char32_t>(c).
            # This warning is external-library noise, not a warning from the langevin library code.
            target_compile_options(gtest PRIVATE -Wno-character-conversion) # silence character-conversion warnings in gtest
        endif()

        add_executable(langevin_tests   # test executable
            ${TEST_SOURCES}
        )

        target_link_libraries(langevin_tests
            PRIVATE
                langevin_core
                GTest::gtest_main
        )

        target_compile_options(langevin_tests
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )

        include(GoogleTest)
        gtest_discover_tests(langevin_tests)

    else()
        message(STATUS "BUILD_TESTING is ON, but no tests/*.cpp files were found.")
    endif()
endif()
