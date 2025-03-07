//
// Created by Zhen Chen on 12/8/24.
//
#include "../include/ExtraEnergyTermsBase.h"

#include <iostream>
#include <tbb/parallel_for.h>

namespace LibShell {

void ExtraEnergyTermsBase::TestFuncGradHessian(
std::function<double(const Eigen::VectorXd &, Eigen::VectorXd *, Eigen::SparseMatrix<double> *, bool)> obj_Func,
const Eigen::VectorXd &x0) {
    Eigen::VectorXd dir = x0;
    dir.setRandom();

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> H;

    double f = obj_Func(x0, &grad, &H, false);
    std::cout << "energy: " << f << ", gradient L2-norm: " << grad.norm() << ", hessian L2-norm: " << H.norm() << std::endl;

    for (int i = 3; i < 10; i++) {
        double eps = std::pow(0.1, i);
        Eigen::VectorXd x = x0 + eps * dir;
        Eigen::VectorXd grad1;
        double f1 = obj_Func(x, &grad1, nullptr, false);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "energy - gradient: " << (f1 - f) / eps - grad.dot(dir) << std::endl;
        std::cout << "gradient - hessian: " << ((grad1 - grad) / eps - H * dir).norm() << std::endl;
    }
}

template <class DerivedA>
void ExtraEnergyTermsBase::proj_sym_matrix(Eigen::MatrixBase<DerivedA>& A, const HessianProjectType& projType) {
    // no projection
    if (projType == HessianProjectType::kNone) {
        return;
    }
    Eigen::SelfAdjointEigenSolver<DerivedA> eigenSolver(A);
    if (eigenSolver.eigenvalues()[0] >= 0) {
        return;
    }

    using T = typename DerivedA::Scalar;
    Eigen::Matrix<T, -1, 1> D = eigenSolver.eigenvalues();
    for (int i = 0; i < A.rows(); ++i) {
        if (D[i] < 0) {
            if (projType == HessianProjectType::kMaxZero) {
                D[i] = 0;
            } else if (projType == HessianProjectType::kAbs) {
                D[i] = -D[i];
            } else {
                D[i] = 0;
            }
        } else {
            break;
        }
    }
    A = eigenSolver.eigenvectors() * D.asDiagonal() * eigenSolver.eigenvectors().transpose();
}

void ExtraEnergyTermsBase::initialization(const Eigen::MatrixXd& rest_pos,
                                      const MeshConnectivity& mesh,
                                      double youngs,
                                      double shear,
                                      double thickness,
                                      double poisson,
                                      int quad_oder) {
    m_youngs = youngs;
    m_shear = shear;
    m_thickness = thickness;
    m_poisson = poisson;
    m_quad_points = build_quadrature_points(quad_oder);

    m_face_area.resize(mesh.nFaces());
    for (int f = 0; f < mesh.nFaces(); f++) {
        Eigen::Vector3d e1 = rest_pos.row(mesh.faceVertex(f, 1)) - rest_pos.row(mesh.faceVertex(f, 0));
        Eigen::Vector3d e2 = rest_pos.row(mesh.faceVertex(f, 2)) - rest_pos.row(mesh.faceVertex(f, 0));
        m_face_area[f] = 0.5 * e1.cross(e2).norm();
    }
}

double ExtraEnergyTermsBase::compute_thirdFundamentalForm_energy(const Eigen::MatrixXd& cur_pos,
                                                                 const Eigen::VectorXd& edge_dofs,
                                                                 const MeshConnectivity& mesh,
                                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                                 Eigen::VectorXd* deriv,
                                                                 std::vector<Eigen::Triplet<double>>* hessian,
                                                                 bool is_proj) {
    return 0;
}

double ExtraEnergyTermsBase::compute_thirdFundamentalForm_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                                 const Eigen::VectorXd& edge_dofs,
                                                                 const MeshConnectivity& mesh,
                                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                                 int face,
                                                                 Eigen::VectorXd* deriv,
                                                                 Eigen::MatrixXd* hessian,
                                                                 bool is_proj) {
    return 0;
}

void ExtraEnergyTermsBase::test_compute_thirdFundamentalForm_energy(const MeshConnectivity& mesh,
                                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::MatrixXd& cur_pos,
                                                                 const Eigen::VectorXd& edge_dofs) {
    // Test the third fundamental form energy computation
}

void ExtraEnergyTermsBase::test_compute_thirdFundamentalForm_energy_perface(const MeshConnectivity& mesh,
                                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::MatrixXd& cur_pos,
                                                                 const Eigen::VectorXd& edge_dofs,
                                                                 int face) {
    // Test the third fundamental form energy computation per face
}



template void ExtraEnergyTermsBase::proj_sym_matrix(Eigen::MatrixBase<Eigen::Matrix<double, 3, 3>>& symA, const HessianProjectType& projType);
template void ExtraEnergyTermsBase::proj_sym_matrix(Eigen::MatrixBase<Eigen::MatrixXd>& symA, const HessianProjectType& projType);

}  // namespace LibShell