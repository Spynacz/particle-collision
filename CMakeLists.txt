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

# --- Shared Sources ---
# These files are used by BOTH the GUI and the Benchmark
set(COMMON_SOURCES
    src/Particle.cpp
    src/gpu_physics.cu
    src/ColorGradient.cpp
)

# ==========================================
# TARGET 1: The GUI Application
# ==========================================
add_executable(particles-gui src/main.cpp ${COMMON_SOURCES})

set_property(TARGET particles-gui PROPERTY CXX_STANDARD 17)
set_property(TARGET particles-gui PROPERTY CUDA_STANDARD 14)

target_link_libraries(particles-gui PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# ==========================================
# TARGET 2: The Benchmark Tool
# ==========================================
add_executable(particles-bench src/bench.cpp ${COMMON_SOURCES})

set_property(TARGET particles-bench PROPERTY CXX_STANDARD 17)
set_property(TARGET particles-bench PROPERTY CUDA_STANDARD 14)

# Note: We still link SFML Graphics because Particle.hpp uses sf::Color,
# even if we don't open a window.
target_link_libraries(particles-bench PRIVATE
    sfml-graphics
    sfml-window
    sfml-system
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# --- Installation & Windows DLLs ---
install(TARGETS particles-gui particles-bench RUNTIME DESTINATION bin)

if(WIN32)
    # Copy OpenAL to the bin folder so the exe can find it
    add_custom_command(
        TARGET particles-gui
        COMMENT "Copy OpenAL DLL"
        PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy 
        ${SFML_SOURCE_DIR}/extlibs/bin/$<IF:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>,x64,x86>/openal32.dll 
        $<TARGET_FILE_DIR:particles-gui>
        VERBATIM)
endif()
