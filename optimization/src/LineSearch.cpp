
#include "../include/LineSearch.h"

#include <iostream>

namespace OptSolver {
// Backtracking line search with Armijo condition
double BacktrackingArmijo(
    const Eigen::VectorXd &x, const Eigen::VectorXd &grad,
    const Eigen::VectorXd &dir,
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *,
                         Eigen::SparseMatrix<double> *, bool)>
        obj_func,
    const double alpha_init) {
  const double c = 0.2;
  const double rho = 0.5;
  double alpha = alpha_init;

  Eigen::VectorXd xNew = x + alpha * dir;
  double fNew = obj_func(xNew, nullptr, nullptr, false);
  double f = obj_func(x, nullptr, nullptr, false);
  const double cache = c * grad.dot(dir);

  while (fNew > f + alpha * cache) {
    alpha *= rho;
    xNew = x + alpha * dir;
    fNew = obj_func(xNew, nullptr, nullptr, false);
  }

  return alpha;
}
} // namespace OptSolver
