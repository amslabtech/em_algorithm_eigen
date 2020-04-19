cmake_minimum_required(VERSION 3.0)

project(em_algorithm_eigen)

set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O2")

find_package(Eigen3 REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
)

add_library(em_algorithm_eigen src/em_algorithm_eigen.cpp)

if(EXISTS ${PROJECT_SOURCE_DIR}/pybind11)
    message("pybind11 exists")
    add_subdirectory(pybind11)
    pybind11_add_module(em_algorithm_eigen_py SHARED src/em_algorithm_eigen_py.cpp src/em_algorithm_eigen.cpp)
else()
    message("pybind11 does not exists")
endif()

install(TARGETS em_algorithm_eigen DESTINATION lib)
install(FILES include/${PROJECT_NAME}/em_algorithm_eigen.h DESTINATION include/${PROJECT_NAME})
