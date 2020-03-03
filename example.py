#!/usr/bin/env python

import em_algorithm_eigen_py

import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt

def ex1():
    em = em_algorithm_eigen_py.EMAlgorithm()
    x = np.array([0, 0])
    mu = np.array([0, 0])
    sigma = np.array([[1, 0],
                      [0, 1]])
    v = em.get_normal_distribution_value(x, mu, sigma)
    print(v)

def ex2():
    em = em_algorithm_eigen_py.EMAlgorithm()
    N = 1000
    D = 2
    K = 2

    data = np.random.normal(0, 0.3, (N, D))
    data = np.concatenate((data, np.random.normal(1, 0.5, (N, D))))

    print(data.shape)
    em.set_data(data)
    em.set_num_of_distributions(K)
    mu, sigma, pi = em.caluculate(100)
    likelihood = em.get_log_likelihood(data, mu, sigma, pi)
    pprint((mu, sigma, pi))
    print(likelihood)

    if D==2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1], label="raw", s=1)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.stack([X, Y], axis=2)
        Z = list()
        for p in pos:
            Z_ = list()
            for p_ in p:
                z = em.get_gmm_value(p_, mu, sigma, pi)
                Z_.append(z)
            Z.append(Z_)
        ax.contour(X, Y, Z)
        plt.show()

def main():
    ex2()

if __name__=='__main__':
    main()
