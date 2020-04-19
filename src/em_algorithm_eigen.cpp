#include "em_algorithm_eigen/em_algorithm_eigen.h"

namespace em_algorithm_eigen
{

EMAlgorithm::EMAlgorithm(void)
:k(1)
{
}

void EMAlgorithm::set_data(const std::vector<Eigen::VectorXd>& data_)
{
    data = data_;
    n = data.size();
    if(n > 1){
        d = data[0].size();
    }else{
        d = 0;
    }
}

void EMAlgorithm::set_num_of_distributions(unsigned int k_)
{
    k = k_;
}

double EMAlgorithm::get_normal_distribution_value(Eigen::VectorXd x_, Eigen::VectorXd mu_, Eigen::MatrixXd sigma_)
{
    // std::cout << "x:\n" << x_.transpose() << std::endl;
    // std::cout << "mu:\n" << mu_.transpose() << std::endl;
    // std::cout << "sigma:\n" << sigma_ << std::endl;
    d = x_.size();
    return 1.0 / (std::pow(2.0 * M_PI, d / 2.0) * sqrt(sigma_.determinant())) * std::exp(-0.5 * (x_ - mu_).transpose() * sigma_.inverse() * (x_ - mu_));
}

std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>, std::vector<double> > EMAlgorithm::calculate(unsigned int max_loop=100)
{
    const unsigned int N = n - 1;

    std::vector<Eigen::VectorXd> mu(k);
    std::vector<Eigen::MatrixXd> sigma(k);
    for(auto& sigma_ : sigma){
        sigma_ = Eigen::MatrixXd::Identity(d, d);
    }
    std::vector<double> pi(k);
    for(auto& pi_ : pi){
        pi_ = 1.0 / k;
    }
    std::vector<std::vector<double> > gamma(N);
    for(auto& g : gamma){
        g.resize(k);
    }

    // Eigen::VectorXd data_mu = Eigen::VectorXd::Zero(d);
    // for(unsigned int i=0;i<N;i++){
    //     data_mu += data[i];
    // }
    // data_mu /= static_cast<double>(N);
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> dist(0, N);
    for(auto& mu_ : mu){
        mu_ = data[dist(mt)];
    }

    for(unsigned int c=0;c<max_loop;c++){
        // std::cout << "c: " << c << std::endl;
        // E step
        // std::cout << "E step" << std::endl;
        for(unsigned int n_=0;n_<N;n_++){
            for(unsigned int k_=0;k_<k;k_++){
                double sum = 0;
                for(unsigned int k__=0;k__<k;k__++){
                    sum += pi[k__] * get_normal_distribution_value(data[n_], mu[k__], sigma[k__]);
                }
                gamma[n_][k_] = pi[k_] * get_normal_distribution_value(data[n_], mu[k_], sigma[k_]) / sum;
            }
        }
        // M step
        // std::cout << "M step" << std::endl;
        std::vector<Eigen::VectorXd> mu_new(k);
        std::vector<Eigen::MatrixXd> sigma_new(k);
        std::vector<double> pi_new(k);
        for(unsigned int k_=0;k_<k;k_++){
           mu_new[k_] = Eigen::VectorXd::Zero(d);
           sigma_new[k_] = Eigen::MatrixXd::Zero(d, d);
           double gamma_sum = 0;
           // std::cout << "k_: " << k_ << std::endl;
           for(unsigned int n_=0;n_<N;n_++){
               // std::cout << "n_: " << k_ << std::endl;
               mu_new[k_] += gamma[n_][k_] * data[n_];
               // std::cout << mu_new[k_].transpose() << std::endl;
               Eigen::VectorXd dm = data[n_] - mu[k_];
               // std::cout << dm.transpose() << std::endl;
               sigma_new[k_] += gamma[n_][k_] * dm * dm.transpose();
               gamma_sum += gamma[n_][k_];
           }
           // std::cout << "gamma_sum: " << gamma_sum << std::endl;
           mu_new[k_] /= gamma_sum;
           sigma_new[k_] /= gamma_sum;
           pi_new[k_] = gamma_sum / static_cast<double>(N);
        }
        mu = mu_new;
        sigma = sigma_new;
        pi = pi_new;
        // std::cout << "loop end" << std::endl;
    }
    return std::make_tuple(mu, sigma, pi);
}

double EMAlgorithm::get_gmm_value(const Eigen::VectorXd& x, const std::vector<Eigen::VectorXd>& mu_, const std::vector<Eigen::MatrixXd>& sigma_, const std::vector<double>& pi_)
{
    // return value from Gaussian Mixture Model
    double sum = 0;
    for(unsigned int k_=0;k_<k;k_++){
        sum += pi_[k_] * get_normal_distribution_value(x, mu_[k_], sigma_[k_]);
    }
    return sum;
}

double EMAlgorithm::get_log_likelihood(const std::vector<Eigen::VectorXd>& data_, const std::vector<Eigen::VectorXd>& mu_, const std::vector<Eigen::MatrixXd>& sigma_, const std::vector<double>& pi_)
{
    double sum = 0;
    unsigned int n_ = data_.size();
    for(unsigned int i=0;i<n_;i++){
        sum += std::log(get_gmm_value(data_[i], mu_, sigma_, pi_));
    }
    return sum;
}

};
