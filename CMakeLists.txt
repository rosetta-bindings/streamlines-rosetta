# -------------------------------------------------------------------
# CLI configuration:
#   - Native release  : cmake -DCMAKE_BUILD_TYPE=Release ..
#   - Native debug    : cmake -DCMAKE_BUILD_TYPE=Debug ..
#
# Then: make -j12
# -------------------------------------------------------------------
cmake_minimum_required(VERSION 3.28)
project(triax VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# ============================================================================
# FetchContent for dependencies
# ============================================================================
include(FetchContent)

# ----------------------------------------------------------------------------
# TRIAX Library
# ----------------------------------------------------------------------------
# EXCLUDE_FROM_ALL: triax's own CMakeLists only builds executables (triax,
# compact, mixture) that the bindings don't need — we just want the sources
# fetched so the headers can be compiled into each binding.
message(STATUS "Fetching surface-streamlines library...")
FetchContent_Declare(
    streamlines
    GIT_REPOSITORY https://github.com/LookUpGeoscience/surface-streamlines.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/streamlines
    BINARY_DIR ${CMAKE_BINARY_DIR}/extern/streamlines-build
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(streamlines)
FetchContent_GetProperties(streamlines SOURCE_DIR STREAMLINES_INCLUDE_DIR)

# ----------------------------------------------------------------------------
# ROSETTA Library
# ----------------------------------------------------------------------------
message(STATUS "Fetching Rosetta library...")
FetchContent_Declare(
    Rosetta
    GIT_REPOSITORY https://github.com/rosetta-bindings/rosetta.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/rosetta
    BINARY_DIR ${CMAKE_BINARY_DIR}/extern/rosetta-build
)
FetchContent_MakeAvailable(rosetta)
FetchContent_GetProperties(rosetta SOURCE_DIR ROSETTA_SOURCE_DIR)
set(ROSETTA_INCLUDE_DIR "${ROSETTA_SOURCE_DIR}/include")

# The rosetta_gen code generator lives in tools/ and has its
# own standalone CMakeLists.txt — build it as a subdirectory.
# NOTE: rosetta_gen requires CMake >= 3.28.
message(STATUS "Building rosetta_gen tool...")
add_subdirectory(
    ${ROSETTA_SOURCE_DIR}/tools/rosetta_gen
    ${CMAKE_BINARY_DIR}/extern/rosetta_gen-build
)
