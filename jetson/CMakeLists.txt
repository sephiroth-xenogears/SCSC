# ============================================================================
# Phase 7: Full-Body 17-Axis Active Inference — CMake Build
# Platform: Jetson AGX Orin (sm_87)
# ============================================================================
cmake_minimum_required(VERSION 3.22)
project(scsc_phase7 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# ── CUDA Architecture (Jetson AGX Orin = sm_87) ──
set(CMAKE_CUDA_ARCHITECTURES 87)

# ── OpenCV ──
find_package(OpenCV 4.8 REQUIRED COMPONENTS
    core imgproc videoio dnn objdetect imgcodecs highgui)

# ── Sources ──
# Main compilation unit (.cu so nvcc handles everything)
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/main_phase7.cpp)

# CUDA kernel files (header-only .cuh, included from engine)
set(CUDA_HEADERS
    cuda/primitives.cuh
    cuda/gen_model.cuh
    cuda/particle_filter.cuh
    cuda/efe_planner.cuh
    cuda/aif_engine.h
)

# C++ module headers
set(CPP_HEADERS
    include/types.h
    include/body_config.h
    include/spsc_queue.h
    src/camera.h
    src/face_detect.h
    src/observation.h
    src/servo_cmd.h
    src/safety.h
    src/logger.h
)

# ── Build main as CUDA (to link kernels) ──
# We rename .cpp to .cu at build time via a custom command
set(MAIN_CU ${CMAKE_CURRENT_BINARY_DIR}/main_phase7.cu)
add_custom_command(
    OUTPUT ${MAIN_CU}
    COMMAND ${CMAKE_COMMAND} -E copy ${MAIN_SOURCE} ${MAIN_CU}
    DEPENDS ${MAIN_SOURCE}
    COMMENT "Preparing CUDA compilation unit"
)

add_executable(phase7_main ${MAIN_CU} ${CUDA_HEADERS} ${CPP_HEADERS})

target_include_directories(phase7_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(phase7_main PRIVATE
    ${OpenCV_LIBS}
    curand
    pthread
)

# ── NVCC flags ──
target_compile_options(phase7_main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=-Wall,-Wextra
    >
)

# ── Headless build option ──
option(HEADLESS_BUILD "Build without GUI (no highgui)" OFF)
if(HEADLESS_BUILD)
    target_compile_definitions(phase7_main PRIVATE HEADLESS_BUILD)
endif()

# ── Install ──
install(TARGETS phase7_main DESTINATION bin)

# ── Info target ──
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "Project: SCSC Phase 7"
    COMMAND ${CMAKE_COMMAND} -E echo "CUDA Arch: sm_87 (Jetson AGX Orin)"
    COMMAND ${CMAKE_COMMAND} -E echo "OpenCV: ${OpenCV_VERSION}"
    VERBATIM
)
