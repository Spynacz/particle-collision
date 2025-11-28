cmake_minimum_required(VERSION 3.18)

# Enable C++ and CUDA
project(particles-collision LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# --- Dependencies ---
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(SFML COMPONENTS graphics window system REQUIRED)
find_package(MPI REQUIRED)  # <--- Added MPI Dependency

# --- Shared CPU Sources ---
# We moved gpu_physics.cu OUT of here because MPI doesn't need it.
set(COMMON_SOURCES
    src/Particle.cpp
    src/ColorGradient.cpp
)

# ==========================================
# Target 1: Original Benchmark (OpenMP + CUDA)
# ==========================================
add_executable(particles-bench 
    src/main.cpp 
    src/gpu_physics.cu  # CUDA kernel added specifically here
    ${COMMON_SOURCES}
)

set_property(TARGET particles-bench PROPERTY CXX_STANDARD 17)
set_property(TARGET particles-bench PROPERTY CUDA_STANDARD 14)

target_link_libraries(particles-bench PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# ==========================================
# Target 2: MPI Implementation
# ==========================================
add_executable(particles-mpi 
    src/main-mpi.cpp 
    ${COMMON_SOURCES}
)

set_property(TARGET particles-mpi PROPERTY CXX_STANDARD 17)

# Link MPI and SFML (No CUDA needed here)
target_link_libraries(particles-mpi PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    MPI::MPI_CXX
)

# --- Installation ---
install(TARGETS particles-bench particles-mpi RUNTIME DESTINATION bin)
