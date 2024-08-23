#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace OptSolver {
///
/// Backtracking line search with Armijo condition
///
/// @param[in] x          the current point
/// @param[in] grad       the gradient at the current point
/// @param[in] dir        the search direction
/// @param[in] obj_func   the objective function, which takes x as input and
///                       returns the function value, gradient, and hessian, together with a
///                       boolean indicating whether the hessian matrix is PSD projected
/// @param[in] alpha_init the initial step size
///
double BacktrackingArmijo(
    const Eigen::VectorXd &x, const Eigen::VectorXd &grad,
    const Eigen::VectorXd &dir,
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *,
                         Eigen::SparseMatrix<double> *, bool)>
        obj_func,
    const double alpha_init = 1.0);
} // namespace OptSolver