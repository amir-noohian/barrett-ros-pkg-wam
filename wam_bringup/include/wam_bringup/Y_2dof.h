#ifndef Y_MATRIX_HPP_
#define Y_MATRIX_HPP_

#include <cmath>
#include <Eigen/Dense>

// Function to calculate the Y matrix
Eigen::MatrixXd calculate_Y_matrix(const Eigen::VectorXd& theta, const Eigen::VectorXd& thetad, const Eigen::VectorXd& thetadd);
Eigen::MatrixXd calculate_Y_matrix(const Eigen::VectorXd& theta, const Eigen::VectorXd& thetad, const Eigen::VectorXd& thetadd) {
    
    Eigen::MatrixXd Y(2, 12);
    Y.setZero(); // Initialize the matrix with zeros
    double g = 9.81;

    // Using 0-based indexing: theta[1] is theta2, theta[3] is theta4, and similarly for thetad and thetadd
    Y(0, 0) = 2.0 * cos(theta[3]) * thetad[3] * thetad[1] + cos(theta[3]) * pow(thetad[3], 2) + 2.0 * sin(theta[3]) * thetadd[1] + sin(theta[3]) * thetadd[3];
    Y(0, 1) = 2.0 * cos(theta[3]) * thetadd[1] - sin(theta[3]) * pow(thetad[3], 2) - 2.0 * sin(theta[3]) * thetad[3] * thetad[1] + cos(theta[3]) * thetadd[3];
    Y(0, 2) = thetadd[3];
    Y(0, 3) = g * sin(theta[3] + theta[1]);
    Y(0, 4) = -g * cos(theta[3] + theta[1]);
    Y(0, 5) = thetadd[1];
    Y(0, 6) = -cos(theta[1]) * g;
    Y(0, 7) = sin(theta[1]) * g;
    Y(0, 8) = tanh(20 * thetad[1]);
    Y(0, 9) = thetad[1];
    Y(0, 10) = 0.0;
    Y(0, 11) = 0.0;

    Y(1, 0) = sin(theta[3]) * thetadd[1] - cos(theta[3]) * pow(thetad[1], 2);
    Y(1, 1) = sin(theta[3]) * pow(thetad[1], 2) + cos(theta[3]) * thetadd[1];
    Y(1, 2) = thetadd[1] + thetadd[3];
    Y(1, 3) = g * sin(theta[3] + theta[1]);
    Y(1, 4) = -g * cos(theta[3] + theta[1]);
    Y(1, 5) = 0.0;
    Y(1, 6) = 0.0;
    Y(1, 7) = 0.0;
    Y(1, 8) = 0.0;
    Y(1, 9) = 0.0;
    Y(1, 10) = tanh(20 * thetad[3]);
    Y(1, 11) = thetad[3];

    return Y;
}

#endif // Y_MATRIX_HPP_