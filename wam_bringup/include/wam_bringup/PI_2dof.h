//Estimated Pvector, gravity included.
#ifndef PI_VECTOR_HPP_
#define PI_VECTOR_HPP_

#include <Eigen/Dense>
Eigen::Matrix<double, 12, 1> initialize_pi() {
    Eigen::Matrix<double, 12, 1> pi;
    pi << 0.270579387175855,
    -0.208695099581966,
    0.168926562021083,  
    -0.800320239033409,
    -0.176517063849070,
    2.05671186163312,
    0.0261152332313845,
    -2.02765298645556,
    2.56764800019947,
    -0.509431115014757,
    0.643188182123626,
    1.48538954968544;

    return pi;
}

#endif