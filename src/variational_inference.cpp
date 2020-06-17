#include "em_algorithm_eigen/variational_inference.h"

namespace em_algorithm_eigen
{

VariationalInference::VariationalInference(void)
{
    n = 0;
    k = 0;
    d = 0;
}

void VariationalInference::initialize_parameters(const std::vector<Eigen::VectorXd>& data)
{
    n = data.size();
    if(n > 1){
        d = data[0].size();
    }else{
        d = 0;
        return;
    }

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::normal_distribution<> dist(0.0, 1.0);

    m0 = Eigen::VectorXd::Zero(d);
    for(unsigned int i=0;i<d;i++){
        m0(i) = rnd();
    }

    m.clear();
    for(unsigned int i=0;i<k;i++){
        Eigen::VectorXd m_ = Eigen::VectorXd::Zero(d);
        for(unsigned int j=0;j<d;j++){
            m_(i) = rnd();
        }
        m.emplace_back(m_);
    }

    nu0 = d;
    alpha0 = 0.1;
    beta0 = 1.0;
    w0 = Eigen::MatrixXd::Identity(d, d);
}

void VariationalInference::set_num_of_distributions(unsigned int k_)
{
    k = k_;
    alpha = Eigen::VectorXd::Ones(k) * alpha0;
    beta = Eigen::VectorXd::Ones(k) * beta0;
    nu = Eigen::VectorXd::Ones(k) * nu0;
    w = std::vector<Eigen::MatrixXd>(k, w0);
}

double VariationalInference::get_normal_distribution_value(Eigen::VectorXd x_, Eigen::VectorXd mu_, Eigen::MatrixXd sigma_)
{
    // std::cout << "x:\n" << x_.transpose() << std::endl;
    // std::cout << "mu:\n" << mu_.transpose() << std::endl;
    // std::cout << "sigma:\n" << sigma_ << std::endl;
    d = x_.size();
    return 1.0 / (std::pow(2.0 * M_PI, d / 2.0) * sqrt(sigma_.determinant())) * std::exp(-0.5 * (x_ - mu_).transpose() * sigma_.inverse() * (x_ - mu_));
}

std::tuple<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>, std::vector<double> > VariationalInference::calculate(const std::vector<Eigen::VectorXd>& data, unsigned int start_k=10, unsigned int max_loop=100)
{
    initialize_parameters(data);
    set_num_of_distributions(start_k);

    // mixture ratio
    std::vector<double> pi(k, 1 / static_cast<double>(k));

    std::vector<Eigen::VectorXd> responsibility = calculate_e_step(data);
    std::vector<Eigen::MatrixXd> sigma = std::vector<Eigen::MatrixXd>(k, Eigen::MatrixXd::Zero(d, d));
    for(unsigned int k_=0;k_<k;k_++){
        sigma[k_] = (nu(k_) * w[k_]).inverse();
    }
    double log_likelihood = get_log_likelihood(data, m, sigma, pi);

    for(unsigned int i=0;i<max_loop;i++){
        calculate_m_step(data, responsibility);
        pi = get_expected_value_of_dirichlet_distribution(alpha);
    }
}

std::vector<Eigen::VectorXd> VariationalInference::calculate_e_step(const std::vector<Eigen::VectorXd>& data)
{
    // E step
    // alpha, beta, nu, m, w -> responsibility
    std::vector<double> pi = get_pi(alpha);

    Eigen::VectorXd lambda_ = Eigen::VectorXd::Zero(k);
    for(unsigned int k_=0;k_<k;k_++){
        double wishart = 0;// \tilde{\lambda}
        for(unsigned int d_=0;d_<d;d_++){
            wishart += boost::math::digamma((nu(k_) + 1 - d_) / 2.0);
        }
        wishart += std::log(2) * d + std::log(w[k_].determinant());
        lambda_(k_) = wishart;
    }

    std::vector<Eigen::VectorXd> log_rho;
    for(unsigned int n_=0;n_<n;n_++){
        log_rho.emplace_back(Eigen::VectorXd::Zero(k));
    }

    for(unsigned int n_=0;n_<n;n_++){
        for(unsigned int k_=0;k_<k;k_++){
            Eigen::VectorXd diff = data[n_] - m[k_];
            log_rho[n_](k_) = pi[k_]
                        + 0.5 * lambda_(k_)
                        - d / (2.0 * beta[k_])
                        - (nu(k_) / 2.0) * diff.transpose() * w[k_] * diff;
        }
    }
    return get_responsibility(log_rho);
}

void VariationalInference::calculate_m_step(const std::vector<Eigen::VectorXd>& data, const std::vector<Eigen::VectorXd>& responsibility)
{
    // M step
    // responsibility -> alpha, beta, nu, m, w
    Eigen::VectorXd n_k = Eigen::VectorXd::Zero(k);
    for(const auto& r : responsibility){
        n_k = n_k + r;
    }
    std::vector<Eigen::VectorXd> x_ = std::vector<Eigen::VectorXd>(k, Eigen::VectorXd::Zero(d));
    for(unsigned k_=0;k_<k;k_++){
        Eigen::VectorXd x_k = Eigen::VectorXd::Zero(d);
        for(unsigned n_=0;n_<n;n_++){
            x_k = x_k + responsibility[n_][k_] * data[n_];
        }
        x_[k_] = x_k / n_k(k_);
    }
    std::vector<Eigen::MatrixXd> s(k, Eigen::MatrixXd::Zero(d, d));
    for(unsigned int k_=0;k_<k;k_++){
        for(unsigned int n_=0;n_<n;n_++){
            Eigen::VectorXd diff = data[n_] - x_[k_];
            s[k_] += responsibility[n_](k_) * diff * diff.transpose();
        }
        s[k_] /= n_k[k_];
    }
    alpha = alpha0 * Eigen::VectorXd::Ones(k) + n_k;
    beta = beta0 * Eigen::VectorXd::Ones(k) + n_k;
    nu = nu0 * Eigen::VectorXd::Ones(k) + n_k;
    for(unsigned int k_=0;k_<k;k_++){
        m[k_] = (beta0 * m0 + n_k[k_] * x_[k_]) / beta(k_);
    }
    for(unsigned int k_=0;k_<k;k_++){
        Eigen::VectorXd diff = x_[k_] - m0;
        w[k_] = w0.inverse()
              + n_k[k_] * s[k_]
              + (beta0 * n_k[k_]) / (beta0 + n_k[k_]) * diff * diff.transpose();
        w[k_] = w[k_].inverse();
    }
}

double VariationalInference::get_gmm_value(const Eigen::VectorXd& x, const std::vector<Eigen::VectorXd>& mu_, const std::vector<Eigen::MatrixXd>& sigma_, const std::vector<double>& pi_)
{
    // return value from Gaussian Mixture Model
    double sum = 0;
    for(unsigned int k_=0;k_<k;k_++){
        sum += pi_[k_] * get_normal_distribution_value(x, mu_[k_], sigma_[k_]);
    }
    return sum;
}

double VariationalInference::get_log_likelihood(const std::vector<Eigen::VectorXd>& data_, const std::vector<Eigen::VectorXd>& mu_, const std::vector<Eigen::MatrixXd>& sigma_, const std::vector<double>& pi_)
{
    double sum = 0;
    unsigned int n_ = data_.size();
    for(unsigned int i=0;i<n_;i++){
        sum += std::log(get_gmm_value(data_[i], mu_, sigma_, pi_));
    }
    return sum;
}

std::vector<double> VariationalInference::get_pi(const Eigen::VectorXd& vec)
{
    double sum = vec.sum();
    std::vector<double> vec_;
    for(unsigned int i=0;i<vec.size();i++){
        vec_.emplace_back(boost::math::digamma(vec(i)) - boost::math::digamma(sum));

    }
    return vec_;
}

std::vector<Eigen::VectorXd> VariationalInference::get_responsibility(const std::vector<Eigen::VectorXd>& log_rho)
{
    std::vector<Eigen::VectorXd> r = log_rho;
    for(unsigned int n_=0;n_<n;n_++){
        r[n_] = (log_rho[n_] - log_sum_exp(log_rho[n_])).array().exp();
    }
    return r;
}

Eigen::VectorXd VariationalInference::log_sum_exp(const Eigen::VectorXd& vec)
{
    double max = vec.maxCoeff();
    Eigen::VectorXd max_vec = Eigen::VectorXd::Constant(vec.size(), max);
    double result = std::log((vec - max_vec).array().exp().sum()) + max;
    return Eigen::VectorXd::Constant(vec.size(), result);
}

std::vector<double> VariationalInference::get_expected_value_of_dirichlet_distribution(const Eigen::VectorXd& alpha_)
{
    std::vector<double> x(k, 0);
    double sum = alpha_.sum();
    for(unsigned int i=0;i<k;i++){
        x[i] = alpha(i) / sum;
    }
    return x;
}

};// em_algorithm_eigen
