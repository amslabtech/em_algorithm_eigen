#include "em_algorithm_eigen/em_algorithm_eigen.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


template <typename T>
using RMatrix = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

namespace py = pybind11;

using namespace em_algorithm_eigen;

PYBIND11_MODULE(em_algorithm_eigen_py, m)
{
    py::class_<EMAlgorithm>(m, "EMAlgorithm")
        .def(py::init<>())
        .def("set_data", &EMAlgorithm::set_data)
        .def("set_num_of_distributions", &EMAlgorithm::set_num_of_distributions)
        .def("caluculate", &EMAlgorithm::caluculate)
        .def("get_normal_distribution_value", &EMAlgorithm::get_normal_distribution_value)
        .def("get_gmm_value", &EMAlgorithm::get_gmm_value)
        .def("get_log_likelihood", &EMAlgorithm::get_log_likelihood);
}
