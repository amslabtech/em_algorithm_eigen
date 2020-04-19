#ifndef __EM_ALGORITHM_EIGEN
#define __EM_ALGORITHM_EIGEN

#include <iostream>
#include <vector>
#include <tuple>
#include <random>

#include <Eigen/Dense>

namespace em_algorithm_eigen
{

class EMAlgorithm
{
public:
    EMAlgorithm(void);

    void set_data(const std::vector<Eigen::VectorXd>&);
    void set_num_of_distributions(unsigned int);
    double get_normal_distribution_value(Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd);
    std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>, std::vector<double> > calculate(unsigned int max_loop);
    double get_gmm_value(const Eigen::VectorXd&, const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::MatrixXd>&, const std::vector<double>&);
    double get_log_likelihood(const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::MatrixXd>&, const std::vector<double>&);

protected:
    unsigned int n;// number of data
    unsigned int k;// number of normal distribution
    unsigned int d;// dimension of data

    std::vector<Eigen::VectorXd> data;
};

};

#endif// __EM_ALGORITHM_EIGEN
