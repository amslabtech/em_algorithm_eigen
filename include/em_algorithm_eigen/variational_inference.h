#ifndef __VARIATIONAL_INFERENCE_H
#define __VARIATIONAL_INFERENCE_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

#include <boost/math/special_functions/digamma.hpp>

#include <Eigen/Dense>

namespace em_algorithm_eigen
{

class VariationalInference
{
public:
    VariationalInference(void);

    void initialize_parameters(const std::vector<Eigen::VectorXd>&);
    void set_num_of_distributions(unsigned int);
    double get_normal_distribution_value(Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd);
    std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>, std::vector<double> > calculate(const std::vector<Eigen::VectorXd>&, unsigned int start_k, unsigned int max_loop);
    std::vector<Eigen::VectorXd> calculate_e_step(const std::vector<Eigen::VectorXd>&);
    void calculate_m_step(const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::VectorXd>&);
    double get_gmm_value(const Eigen::VectorXd&, const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::MatrixXd>&, const std::vector<double>&);
    double get_log_likelihood(const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::MatrixXd>&, const std::vector<double>&);
    std::vector<double> get_pi(const Eigen::VectorXd&);
    std::vector<Eigen::VectorXd> get_responsibility(const std::vector<Eigen::VectorXd>&);
    Eigen::VectorXd log_sum_exp(const Eigen::VectorXd&);
    std::vector<double> get_expected_value_of_dirichlet_distribution(const Eigen::VectorXd&);

protected:
    unsigned int n;// number of data
    unsigned int k;// number of normal distribution
    unsigned int d;// dimension of data
    double alpha0;// initial parameter for dirichlet distribution
    Eigen::VectorXd alpha;// parameter for dirichlet distribution
    double beta0;// initial parameter
    Eigen::VectorXd beta;// parameter
    Eigen::VectorXd m0;// initial mean for gmm
    std::vector<Eigen::VectorXd> m;// mean for gmm
    double nu0;// initial parameter for wishart distribution
    Eigen::VectorXd nu;// parameter for wishart distribution
    Eigen::MatrixXd w0;// initial parameter for wishart distribution
    std::vector<Eigen::MatrixXd> w;// parameter for wishart distribution
};

};// em_algorithm_eigen

#endif// __VARIATIONAL_INFERENCE_H
