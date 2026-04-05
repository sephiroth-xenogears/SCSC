# ============================================================
# SCSC - Super Cybernetics Control System
# CMake Build Configuration
# ============================================================
# Target: Jetson AGX Orin (sm_87, Ampere)
#
# Build modes:
#   Release:  cmake .. -DCMAKE_BUILD_TYPE=Release
#   Debug:    cmake .. -DCMAKE_BUILD_TYPE=Debug
#   Profile:  cmake .. -DCMAKE_BUILD_TYPE=Profile
#
# Profiling (Nsight Compute):
#   cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-G -lineinfo"
#   ncu --set full --output=scsc_kernel ./scsc_binary
# ============================================================

cmake_minimum_required(VERSION 3.18)
project(SCSC LANGUAGES CXX CUDA)

# ---- CUDA Architecture ----
# sm_87 = Jetson AGX Orin (Ampere GA10B)
set(CMAKE_CUDA_ARCHITECTURES 87)

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Build Types ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug flags (for Nsight Compute)
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -lineinfo -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Profile flags (optimized but with debug info)
set(CMAKE_CUDA_FLAGS_PROFILE "-O3 -lineinfo")
set(CMAKE_CXX_FLAGS_PROFILE "-O3 -g")

# ---- Common CUDA Flags ----
# --use_fast_math: enable fast math intrinsics (acceptable for FP32 control)
# --ptxas-options=-v: verbose PTX assembler output (register usage, shared mem)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# ---- Source Files ----
set(CUDA_SOURCES
    src/cuda/active_inference_kernel.cu
)

set(CPP_SOURCES
    src/cuda/main.cu
)

# ---- Header Files ----
set(CUDA_HEADERS
    src/cuda/gen_model.cuh
    src/cuda/active_inference_kernel.cuh
)

# ---- Main Binary ----
add_executable(scsc_binary
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

target_include_directories(scsc_binary PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cuda
)

# ---- Test Binary ----
add_executable(scsc_test
    src/cuda/active_inference_kernel.cu
    src/cuda/test_harness.cu
)

target_include_directories(scsc_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${CMAKE_SOURCE_DIR}/tests
)

# ---- Custom Targets ----

# Run Python reference tests
add_custom_target(test_python
    COMMAND python3 ${CMAKE_SOURCE_DIR}/src/python/test_reference.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python reference tests"
)

# Export test vectors
add_custom_target(export_vectors
    COMMAND python3 ${CMAKE_SOURCE_DIR}/src/python/test_reference.py --export
            ${CMAKE_SOURCE_DIR}/tests/test_vectors
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Exporting test vectors for CUDA verification"
)

# Profile with Nsight Systems
add_custom_target(profile_nsys
    COMMAND nsys profile --output=scsc_baseline $<TARGET_FILE:scsc_binary>
    DEPENDS scsc_binary
    COMMENT "Profiling with Nsight Systems"
)

# Profile with Nsight Compute
add_custom_target(profile_ncu
    COMMAND ncu --set full --output=scsc_kernel $<TARGET_FILE:scsc_binary>
    DEPENDS scsc_binary
    COMMENT "Profiling with Nsight Compute"
)

# Roofline analysis
add_custom_target(profile_roofline
    COMMAND ncu --set roofline --output=scsc_roofline $<TARGET_FILE:scsc_binary>
    DEPENDS scsc_binary
    COMMENT "Roofline analysis with Nsight Compute"
)

# ---- Install ----
install(TARGETS scsc_binary DESTINATION bin)

# ---- Summary ----
message(STATUS "")
message(STATUS "=== SCSC Build Configuration ===")
message(STATUS "  CUDA Architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "================================")
message(STATUS "")
find_package(CUDAToolkit REQUIRED)
target_include_directories(scsc_binary PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(scsc_binary PRIVATE CUDA::cudart)
