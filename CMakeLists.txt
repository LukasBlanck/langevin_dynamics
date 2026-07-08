cmake_minimum_required(VERSION 3.20)

project(langevin_dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# source directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# netcdf
find_package(PkgConfig REQUIRED)
pkg_check_modules(NETCDF REQUIRED IMPORTED_TARGET netcdf)

set(NETCDF_CXX_INCLUDE_DIR /opt/homebrew/opt/netcdf-cxx/include)
set(NETCDF_CXX_LIBRARY /opt/homebrew/opt/netcdf-cxx/lib/libnetcdf-cxx4.dylib)


# core library used by both: langevin_dynamics and langevin_tests
add_library(langevin_core STATIC        # core executable
    ${SOURCE_DIR}/input/input.cpp
)

# hpp files (for core)
target_include_directories(langevin_core    
    PUBLIC
        ${SOURCE_DIR}
        ${NETCDF_CXX_INCLUDE_DIR}
)

# linking netcdf to core
target_link_libraries(langevin_core
    PUBLIC
        PkgConfig::NETCDF
        ${NETCDF_CXX_LIBRARY}
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

    file(GLOB TEST_SOURCES CONFIGURE DEPENDS
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
