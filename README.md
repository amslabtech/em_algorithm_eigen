# em_algorithm_eigen
C++ implementation of EM algorithm using Eigen

## requirement
- Eigen (>= 3)

## build and install
```
$ git clone https://github.com/amslabtech/em_algorithm_eigen.git
$ cd em_algorithm_eigen
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install
# for uninstall
$ cat install_manifest.txt | xargs sudo rm -rf
```
Additionally, we made python bindings of this library using pybind11.
```
$ git clone https://github.com/amslabtech/em_algorithm_eigen.git
$ cd em_algorithm_eigen
$ git submodule update --init --recursive
$ pip install --user .
```
An example for using python bindings is shown in ./example.py