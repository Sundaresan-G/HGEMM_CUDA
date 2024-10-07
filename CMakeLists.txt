cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# V100 - 70, A100 - 80
set(CUDA_COMPUTE_CAPABILITY 70)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Add lineinfo flag
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-G>")

# Add -g flag
# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-g>")


# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(hgemm hgemm.cu ${SRC})
set_target_properties(hgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(hgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})