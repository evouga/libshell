#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

namespace OptSolver {
///
/// Newton solver with line search
///
/// @param[in] obj_func                 the objective function, which takes x as
///                                     input and returns the *function value, gradient,
///                                     and hessian, together with a boolean
///                                     indicating whether the hessian matrix is
///                                     PSD projected
/// @param[in] find_max_step            the function to *find the maximum step
///                                     size, which takes x, direction, and returns the maximum step size
/// @param[in] x0                       the initial guess
/// @param[in] num_iter                 the maximum number of iterations
/// @param[in] grad_tol                 the termination tolerance of the gradient
/// @param[in] x_tol                    the tolerance of the solution
/// @param[in] f_tol                    the tolerance of the function value
/// @param[in] is_proj                  whether to project the hessian matrix to PSD
/// @param[in] display_info             whether to display the information
///
void NewtonSolver(
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *,
                         Eigen::SparseMatrix<double> *, bool)>
        obj_func,
    std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)>
        find_max_step,
    Eigen::VectorXd &x0, int num_iter = 1000,
    double grad_tol = 1e-14, double x_tol = 0, double f_tol = 0,
    bool is_proj = false, bool display_info = false);

///
/// Test the function gradient and hessian
///
/// @param[in] obj_func the objective function, which takes x as input and
///                     returns the function value, gradient, and hessian,
///                     together with a boolean indicating whether the hessian
///                     matrix is PSD projected
/// @param[in] x0       the initial guess
///
void TestFuncGradHessian(
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *,
                         Eigen::SparseMatrix<double> *, bool)>
        obj_func,
    const Eigen::VectorXd &x0);
} // namespace OptSolver