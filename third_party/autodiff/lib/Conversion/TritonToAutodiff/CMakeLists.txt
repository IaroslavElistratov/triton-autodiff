# Explanation
#   if(NOT CMAKE_BUILD_TYPE): Checks if the CMAKE_BUILD_TYPE variable is not already set. This variable determines the build configuration (e.g., Debug, Release).
#   set(CMAKE_BUILD_TYPE Debug ...): Sets CMAKE_BUILD_TYPE to Debug if it hasn’t been specified. The CACHE STRING part makes it a cached variable that can be overridden by the user, and FORCE ensures it takes effect in the cache.
#   The Debug build type is a standard CMake configuration that disables optimizations by default (e.g., -O0) and includes debug symbols (e.g., -g).
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)

add_triton_library(TritonToAutodiff
    TritonToAutodiffPass.cpp
    Utils.cpp
    UtilsIO.cpp

    DEPENDS
    TritonToAutodiffPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    AutodiffIR
)
