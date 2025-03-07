//
// Created by Zhen Chen on 12/8/24.
//
#pragma once

#include "QuadPoints.h"
#include "MeshConnectivity.h"
#include "../include/types.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace LibShell {
class ExtraEnergyTermsBase {
public:
    ExtraEnergyTermsBase() {
        m_quad_points.clear();
        m_face_area.clear();
    }
    virtual ~ExtraEnergyTermsBase() = default;

    void set_quad_points(int quad_order) {
        m_quad_points.clear();
        m_quad_points = build_quadrature_points(quad_order);
    }

    void initialization(const Eigen::MatrixXd& rest_pos,
                        const MeshConnectivity& mesh,
                        double youngs,
                        double shear,
                        double thickness,
                        double poisson,
                        int quad_oder);

    template <class DerivedA>
    void proj_sym_matrix(Eigen::MatrixBase<DerivedA>& A, const HessianProjectType& projType);

    void TestFuncGradHessian(
        std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> obj_Func,
        const Eigen::VectorXd& x0);

    // Y h / 4 * int_U (m * m - 1)^2 dA
    virtual double compute_magnitude_compression_energy(const Eigen::VectorXd& edge_dofs,
                                                        const MeshConnectivity& mesh,
                                                        Eigen::VectorXd* deriv,
                                                        std::vector<Eigen::Triplet<double>>* hessian,
                                                        bool is_proj) = 0;

    virtual double compute_magnitude_compression_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                const MeshConnectivity& mesh,
                                                                int face,
                                                                Eigen::VectorXd* deriv,
                                                                Eigen::MatrixXd* hessian,
                                                                bool is_proj) = 0;

    virtual void test_compute_magnitude_compression_energy(const MeshConnectivity& mesh,
                                                           const Eigen::VectorXd& edge_dofs) = 0;
    virtual void test_compute_magnitude_compression_energy_perface(const MeshConnectivity& mesh,
                                                                   const Eigen::VectorXd& edge_dofs,
                                                                   int face) = 0;

    // mu h / 2 int_U ||n^T dr||^2 dA
    virtual double compute_vector_perp_tangent_energy(const Eigen::MatrixXd& cur_pos,
                                                      const Eigen::VectorXd& edge_dofs,
                                                      const MeshConnectivity& mesh,
                                                      const std::vector<Eigen::Matrix2d>& abars,
                                                      Eigen::VectorXd* deriv,
                                                      std::vector<Eigen::Triplet<double>>* hessian,
                                                      bool is_proj) = 0;
    virtual double compute_vector_perp_tangent_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                              const Eigen::VectorXd& edge_dofs,
                                                              const MeshConnectivity& mesh,
                                                              const std::vector<Eigen::Matrix2d>& abars,
                                                              int face,
                                                              Eigen::VectorXd* deriv,
                                                              Eigen::MatrixXd* hessian,
                                                              bool is_proj) = 0;
    virtual void test_compute_vector_perp_tangent_energy(const MeshConnectivity& mesh,
                                                         const std::vector<Eigen::Matrix2d>& abars,
                                                         const Eigen::MatrixXd& cur_pos,
                                                         const Eigen::VectorXd& edge_dofs) = 0;
    virtual void test_compute_vector_perp_tangent_energy_perface(const MeshConnectivity& mesh,
                                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::MatrixXd& cur_pos,
                                                                 const Eigen::VectorXd& edge_dofs,
                                                                 int face) = 0;

    // mu * h^3 / 24 * int_U ||d(m*m)||^2 dA
    virtual double compute_magnitude_sq_change_energy(const Eigen::VectorXd& edge_dofs,
                                                      const MeshConnectivity& mesh,
                                                      const std::vector<Eigen::Matrix2d>& abars,
                                                      Eigen::VectorXd* deriv,
                                                      std::vector<Eigen::Triplet<double>>* hessian,
                                                      bool is_proj) = 0;
    virtual double compute_magnitude_sq_change_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                              const MeshConnectivity& mesh,
                                                              const std::vector<Eigen::Matrix2d>& abars,
                                                              int face,
                                                              Eigen::VectorXd* deriv,
                                                              Eigen::MatrixXd* hessian,
                                                              bool is_proj) = 0;

    virtual void test_compute_magnitude_sq_change_energy(const MeshConnectivity& mesh,
                                                         const std::vector<Eigen::Matrix2d>& abars,
                                                         const Eigen::VectorXd& edge_dofs) = 0;
    virtual void test_compute_magnitude_sq_change_energy_perface(const MeshConnectivity& mesh,
                                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::VectorXd& edge_dofs,
                                                                 int face) = 0;

    // h^5 / 320 int_U ||Ibar^{-1}III||_SV^2 dA
    virtual double compute_thirdFundamentalForm_energy(const Eigen::MatrixXd& cur_pos,
                                                        const Eigen::VectorXd& edge_dofs,
                                                        const MeshConnectivity& mesh,
                                                        const std::vector<Eigen::Matrix2d>& abars,
                                                        Eigen::VectorXd* deriv,
                                                        std::vector<Eigen::Triplet<double>>* hessian,
                                                        bool is_proj) = 0;
    virtual double compute_thirdFundamentalForm_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                        const Eigen::VectorXd& edge_dofs,
                                                        const MeshConnectivity& mesh,
                                                        const std::vector<Eigen::Matrix2d>& abars,
                                                        int face,
                                                        Eigen::VectorXd* deriv,
                                                        Eigen::MatrixXd* hessian,
                                                        bool is_proj) = 0;
    virtual void test_compute_thirdFundamentalForm_energy(const MeshConnectivity& mesh,
                                                  const std::vector<Eigen::Matrix2d>& abars,
                                                  const Eigen::MatrixXd& cur_pos,
                                                  const Eigen::VectorXd& edge_dofs) = 0;
    virtual void test_compute_thirdFundamentalForm_energy_perface(const MeshConnectivity& mesh,
                                                  const std::vector<Eigen::Matrix2d>& abars,
                                                  const Eigen::MatrixXd& cur_pos,
                                                  const Eigen::VectorXd& edge_dofs,
                                                  int face) = 0;


    // TODO: Change this to protected
public:
    std::vector<QuadraturePoint> m_quad_points;
    std::vector<double> m_face_area;
    double m_youngs;
    double m_shear;
    double m_thickness;
    double m_poisson;
};
}  // namespace LibShell
