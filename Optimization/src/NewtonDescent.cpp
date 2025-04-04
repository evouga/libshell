#include <fstream>
#include <iomanip>

#include <Eigen/Sparse>
#include <spdlog/spdlog.h>

#include "../include/LineSearch.h"
#include "../include/NewtonDescent.h"
#include "../include/Timer.h"

namespace OptSolver {
// Newton solver with line search
void NewtonSolver(
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *, Eigen::SparseMatrix<double> *, bool)> obj_func,
    std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)> find_max_step,
    Eigen::VectorXd &x0,
    int num_iter,
    double grad_tol,
    double x_tol,
    double f_tol,
    bool is_proj_hess,
    bool display_info,
    bool is_swap) {
    const int DIM = x0.rows();
    // Eigen::VectorXd randomVec = x0;
    // randomVec.setRandom();
    // x0 += 1e-6 * randomVec;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
    Eigen::SparseMatrix<double> hessian;

    Eigen::VectorXd neg_grad, delta_x;
    double max_step_size = 1.0;
    double reg = 1e-8;

    bool is_proj = is_proj_hess;
    Timer<std::chrono::high_resolution_clock> total_timer;
    double total_assembling_time = 0;
    double total_solving_time = 0;
    double total_linesearch_time = 0;

    total_timer.start();
    std::ofstream optInfo;
    spdlog::set_level(spdlog::level::info);

    if (display_info) {
        spdlog::set_level(spdlog::level::debug);
    }

    if (display_info) {
        spdlog::debug(
            "Termination Creteria, gradient tolerance: {}, function update tolerance: {}, variable update tolerance: "
            "{}, maximum iteration: {}\n",
            grad_tol, f_tol, x_tol, num_iter);
    }
    int i = 0;

    double f = obj_func(x0, nullptr, nullptr, false);
    if (f == 0) {
        spdlog::info("energy = 0, return");
    }

    Eigen::SparseMatrix<double> I(DIM, DIM);
    I.setIdentity();

    bool is_small_perturb_needed = false;

    for (; i < num_iter; i++) {
        if (display_info) {
            spdlog::debug("iter: {}, ||x||: {}", i, x0.norm());
        }

        Timer<std::chrono::high_resolution_clock> local_timer;
        local_timer.start();
        double f = obj_func(x0, &grad, &hessian, is_proj);
        local_timer.stop();
        double localAssTime = local_timer.elapsed<std::chrono::milliseconds>() * 1e-3;
        total_assembling_time += localAssTime;

        local_timer.start();
        Eigen::SparseMatrix<double> H = hessian;
        spdlog::debug("num of nonzeros: {}, rows: {}, cols: {}, Sparsity: {}%", H.nonZeros(), H.rows(), H.cols(),
                      H.nonZeros() * 100.0 / (H.rows() * H.cols()));

        if (is_small_perturb_needed && is_proj) {
            // due to the numerical issue, we may need to add a small perturbation to
            // the PSD projected hessian matrix
            H += reg * I;
        }

        Eigen::SparseMatrix<double> HT = H.transpose();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(H);

        while (solver.info() != Eigen::Success) {
            if (display_info) {
                if (is_proj) {
                    spdlog::debug("some small perturb is needed to remove round-off error, current reg = {}", reg);
                }

                else {
                    spdlog::debug("Matrix is not positive definite, current reg = {}", reg);
                }
            }

            if (is_proj) {
                is_small_perturb_needed = true;
            }

            H = hessian + reg * I;
            solver.compute(H);
            reg = std::max(2 * reg, 1e-16);

            // if (reg > 1e4 && is_proj_hess) {
            //     spdlog::debug("reg is too large, use SPD hessian instead.");
            //     reg = 1e-6;
            //     is_proj = true;
            //     f = obj_func(x0, &grad, &hessian, is_proj);
            // }
        }

        neg_grad = -grad;
        delta_x = solver.solve(neg_grad);

        local_timer.stop();
        double local_solving_time = local_timer.elapsed<std::chrono::milliseconds>() * 1e-3;
        total_solving_time += local_solving_time;

        max_step_size = find_max_step(x0, delta_x);

        local_timer.start();
        double rate = BacktrackingArmijo(x0, grad, delta_x, obj_func, max_step_size);
        local_timer.stop();
        double local_linesearch_time = local_timer.elapsed<std::chrono::milliseconds>() * 1e-3;
        total_linesearch_time += local_linesearch_time;

        if (!is_proj) {
            reg *= 0.5;
            reg = std::max(reg, 1e-16);
        } else {
            reg = 1e-8;
        }

        if(delta_x.hasNaN()) {
            spdlog::error("Descent direction has nan! Terminate the solver...");
            return;
        }

        x0 = x0 + rate * delta_x;

        double fnew = obj_func(x0, &grad, nullptr, is_proj);

        if (display_info) {
            spdlog::debug("line search rate : {}, actual hessian : {}, reg = {}", rate, !is_proj, reg);
            spdlog::debug("f_old: {}, f_new: {}, grad norm: {}, newton dec: {}, delta_x : {}, delta_f: {}", f, fnew, grad.norm(),
                          delta_x.norm(), rate * delta_x.norm(), f - fnew);
            spdlog::debug("timing info (in total seconds): ");
            spdlog::debug("assembling took: {}, LLT solver took: {}, line search took: {}\n", total_assembling_time,
                          total_solving_time, total_linesearch_time);
        }

        double switch_tol = 1e-4;

        // switch to the actual hessian when close to convergence
        if (is_swap) {
            // this is just some experience value, you can change it
            if (delta_x.norm() < switch_tol) {
                is_proj = false;
            }
        }
        

        // Termination conditions
        if (rate < 1e-8) {
            spdlog::info("terminate with small line search rate (<1e-8): L2-norm = {}", grad.norm());
            break;
        }

        if (grad.norm() < grad_tol) {
            spdlog::info("terminate with gradient L2-norm = {}", grad.norm());
            break;
        }

        if (rate * delta_x.norm() < x_tol) {
            spdlog::info("terminate with small variable change, gradient L2-norm = {}", grad.norm());
            break;
        }

        if (f - fnew < f_tol) {
            spdlog::info("terminate with small energy change, gradient L2-norm = {}", grad.norm());
            break;
        }
    }

    if (i >= num_iter) {
        spdlog::info("terminate with reaching the maximum iteration, with gradient L2-norm = {}", grad.norm());
    }

    f = obj_func(x0, &grad, nullptr, false);
    spdlog::info("end up with energy: {}, gradient: {}", f, grad.norm());

    total_timer.stop();
    if (display_info) {
        spdlog::info(
            "total time costed (s): {}, within that, assembling took: {}, LLT solver took: {}, line search took: {}",
            total_timer.elapsed<std::chrono::milliseconds>() * 1e-3, total_assembling_time, total_solving_time,
            total_linesearch_time);
    }
}

// Test the function gradient and hessian
void TestFuncGradHessian(
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *, Eigen::SparseMatrix<double> *, bool)> obj_Func,
    const Eigen::VectorXd &x0) {
    Eigen::VectorXd dir = x0;
    dir.setRandom();

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> H;

    double f = obj_Func(x0, &grad, &H, false);
    spdlog::info("energy: {}, gradient L2-norm: {}", f, grad.norm());
    if (f == 0) return;

    for (int i = 3; i < 10; i++) {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd x = x0 + eps * dir;
        Eigen::VectorXd grad1;
        double f1 = obj_Func(x, &grad1, nullptr, false);

        spdlog::info("eps: {}", eps);
        spdlog::info("energy - gradient : {}", (f1 - f) / eps - grad.dot(dir));
        spdlog::info("energy - hessian : {}", ((grad1 - grad) / eps - H * dir).norm());
    }
}
}  // namespace OptSolver