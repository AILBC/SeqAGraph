cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(proj_name index_elemwise_op)
project(${proj_name} CXX)

set(CUDA_ENABLE true)
if (CUDA_ENABLE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code\"=sm_86,compute_86\"")
endif()

set(CMAKE_CXX_STANDARD 14)
set (CMAKE_VS_JUST_MY_CODE_DEBUGGING 1)

set(TORCH_INCLUDE1 E:/python/libtorch/include)
set(TORCH_INCLUDE2 E:/python/libtorch/include/torch/csrc/api/include)
set(PYTHON_INCLUDE E:/python/include)

set(TORCH_LIBRARIES E:/python/libtorch/lib)
set(PYTHON_LIBRARIES E:/python/libs)

include_directories(
${TORCH_INCLUDE1}
${TORCH_INCLUDE2}
${PYTHON_INCLUDE}
)

link_directories(
${TORCH_LIBRARIES}
${PYTHON_LIBRARIES}
)

file(GLOB head_files "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cuh")
file(GLOB src_files "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda/*.cu")

list(APPEND head_list ${head_files})
list(APPEND src_list ${src_files})

message(STATUS "Pytorch status:")
message(STATUS "libraries: ${TORCH_LIBRARIES}")
message(STATUS "project_source_dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "cmake_source_dir: ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${proj_name} ${src_list} ${head_list})

file(GLOB torch_lib_files "${TORCH_LIBRARIES}/*.lib")
file(GLOB python_lib_files "${PYTHON_LIBRARIES}/*.lib")

list(APPEND torch_lib_list ${torch_lib_files})
list(APPEND python_lib_list ${python_lib_files})


target_link_libraries(${proj_name}
${torch_lib_files}
${python_lib_files}
)