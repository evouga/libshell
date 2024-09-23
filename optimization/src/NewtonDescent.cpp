#include <fstream>
#include <iomanip>

#include <Eigen/Sparse>

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

    if (display_info) {
        std::cout << "============= Termination Creteria ============="
                  << "\ngradient tolerance: " << grad_tol
                  << "\nfunction update tolerance: " << f_tol
                  << "\nvariable update tolerance: " << x_tol
                  << "\nmaximum iteration: " << num_iter
                  << "\n==============================================\n"
                  << std::endl;
    }
    int i = 0;

    double f = obj_func(x0, &grad, nullptr, false);
    if (grad.norm() < grad_tol) {
        std::cout << "initial gradient norm = " << grad.norm() << ", is smaller than the gradient tolerance: " << grad_tol << ", return" << std::endl;
        return;
    }

    Eigen::SparseMatrix<double> I(DIM, DIM);
    I.setIdentity();

    bool is_small_perturb_needed = false;

    for (; i < num_iter; i++) {
        if (display_info) {
            std::cout << "\niteration: " << i << std::endl;
        }

        Timer<std::chrono::high_resolution_clock> local_timer;
        local_timer.start();
        double f = obj_func(x0, &grad, &hessian, is_proj);
        local_timer.stop();
        double localAssTime = local_timer.elapsed<std::chrono::milliseconds>() * 1e-3;
        total_assembling_time += localAssTime;

        local_timer.start();
        Eigen::SparseMatrix<double> H = hessian;
        std::cout << "num of nonzeros: " << H.nonZeros() << ", rows: " << H.rows() << ", cols: " << H.cols()
                  << ", Sparsity: " << H.nonZeros() * 100.0 / (H.rows() * H.cols()) << "%" << std::endl;

        Eigen::SparseMatrix<double> HT = H.transpose();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(H);

        while (solver.info() != Eigen::Success) {
            if (display_info) {
                if (is_proj) {
                    std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg
                              << std::endl;
                }

                else {
                    std::cout << "the hessian matrix is not SPD, add reg * I to make it PSD, current reg = " << reg
                              << std::endl;
                }
            }

            if (is_proj) {
                is_small_perturb_needed = true;
            }

            H = hessian + reg * I;
            solver.compute(H);
            reg = std::max(2 * reg, 1e-16);

            if (reg > 1e4 && is_proj_hess) {
                // the actual hessian is far from SPD, switch to the PSD hessian if enabled by the user
                // Notice that 1e4 is just some experience value, you can change it
                if(!is_proj) {
                    std::cout << "reg is too large, use SPD hessian instead." << std::endl;
                    reg = 1e-6;
                    is_proj = true;
                    f = obj_func(x0, &grad, &hessian, is_proj);
                } else {
                    std::cout << "reg is too large to get rid of round-off error in the PSD hessian. Please check your implementation" << std::endl;
                    return;
                }
            }
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

        x0 = x0 + rate * delta_x;

        double fnew = obj_func(x0, &grad, nullptr, is_proj);
        if (display_info) {
            std::cout << "line search rate : " << rate << ", actual hessian : " << !is_proj << ", reg = " << reg
                      << std::endl;
            std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm()
                      << ", delta x: " << rate * delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
            std::cout << "timing info (in total seconds): " << std::endl;
            std::cout << "assembling took: " << total_assembling_time << ", LLT solver took: " << total_solving_time
                      << ", line search took: " << total_linesearch_time << std::endl;
        }

        // switch to the actual hessian when close to convergence
        if (is_swap) {
            // this is just some experience value, you can change it
            if ((f - fnew) / f < 1e-5 || delta_x.norm() < 1e-5 || grad.norm() < 1e-4) {
                is_proj = false;
            }
        }
        

        // Termination conditions
        if (rate < 1e-8) {
            std::cout << "terminate with small line search rate (<1e-8): L2-norm = " << grad.norm() << std::endl;
            break;
        }

        if (grad.norm() < grad_tol) {
            std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
            break;
        }

        if (rate * delta_x.norm() < x_tol) {
            std::cout << "terminate with small variable change (<1e-8): L2-norm = " << grad.norm() << std::endl;
            break;
        }

        if (f - fnew < f_tol) {
            std::cout << "terminate with small energy change (<1e-8): L2-norm = " << grad.norm() << std::endl;
            break;
        }
    }

    if (i >= num_iter) {
        std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;
    }

    f = obj_func(x0, &grad, nullptr, false);
    std::cout << "end up with energy: " << f << ", gradient: " << grad.norm() << std::endl;

    total_timer.stop();
    if (display_info) {
        std::cout << "total time costed (s): " << total_timer.elapsed<std::chrono::milliseconds>() * 1e-3
                  << ", within that, assembling took: " << total_assembling_time
                  << ", LLT solver took: " << total_solving_time << ", line search took: " << total_linesearch_time
                  << std::endl;
    }
}

// Test the function gradient and hessian
void TestFuncGradHessian(
    std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *, Eigen::SparseMatrix<double> *, bool)> obj_Func,
    const Eigen::VectorXd &x0) {
    Eigen::VectorXd dir = x0;
    dir(0) = 0;
    dir.setRandom();

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> H;

    double f = obj_Func(x0, &grad, &H, false);
    std::cout << "energy: " << f << ", gradient L2-norm: " << grad.norm() << std::endl;
    if (f == 0) return;

    for (int i = 3; i < 10; i++) {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd x = x0 + eps * dir;
        Eigen::VectorXd grad1;
        double f1 = obj_Func(x, &grad1, nullptr, false);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "energy - gradient : " << (f1 - f) / eps - grad.dot(dir) << std::endl;
        std::cout << "energy - hessian : " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
    }
}
}  // namespace OptSolver
