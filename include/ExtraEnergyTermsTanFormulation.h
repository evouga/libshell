//
// Created by Zhen Chen on 12/8/24.
//

#pragma once

#include "ExtraEnergyTermsBase.h"
#include "../include/MidedgeAngleGeneralTanFormulation.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace LibShell {
class ExtraEnergyTermsTanFormulation : public ExtraEnergyTermsBase {
public:
    // Y h / 4 * int_U (m * m - 1)^2 dA
    double compute_magnitude_compression_energy(const Eigen::VectorXd& edge_dofs,
                                                const MeshConnectivity& mesh,
                                                Eigen::VectorXd* deriv,
                                                std::vector<Eigen::Triplet<double>>* hessian,
                                                bool is_proj) override;

    double compute_magnitude_compression_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                        const MeshConnectivity& mesh,
                                                        int face,
                                                        Eigen::VectorXd* deriv,
                                                        Eigen::MatrixXd* hessian,
                                                        bool is_proj) override;

    void test_compute_magnitude_compression_energy(const MeshConnectivity& mesh,
                                                   const Eigen::VectorXd& edge_dofs) override;
    void test_compute_magnitude_compression_energy_perface(const MeshConnectivity& mesh,
                                                           const Eigen::VectorXd& edge_dofs,
                                                           int face) override;

    // mu h / 2 int_U ||n^T dr||^2 dA
    double compute_vector_perp_tangent_energy(const Eigen::MatrixXd& cur_pos,
                                              const Eigen::VectorXd& edge_dofs,
                                              const MeshConnectivity& mesh,
                                              const std::vector<Eigen::Matrix2d>& abars,
                                              Eigen::VectorXd* deriv,
                                              std::vector<Eigen::Triplet<double>>* hessian,
                                              bool is_proj) override;
    double compute_vector_perp_tangent_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                      const Eigen::VectorXd& edge_dofs,
                                                      const MeshConnectivity& mesh,
                                                      const std::vector<Eigen::Matrix2d>& abars,
                                                      int face,
                                                      Eigen::VectorXd* deriv,
                                                      Eigen::MatrixXd* hessian,
                                                      bool is_proj) override;
    void test_compute_vector_perp_tangent_energy(const MeshConnectivity& mesh,
                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                 const Eigen::MatrixXd& cur_pos,
                                                 const Eigen::VectorXd& edge_dofs) override;
    void test_compute_vector_perp_tangent_energy_perface(const MeshConnectivity& mesh,
                                                         const std::vector<Eigen::Matrix2d>& abars,
                                                         const Eigen::MatrixXd& cur_pos,
                                                         const Eigen::VectorXd& edge_dofs,
                                                         int face) override;

    // mu * h^3 / 24 * int_U ||d(m*m)||^2 dA
    double compute_magnitude_sq_change_energy(const Eigen::VectorXd& edge_dofs,
                                              const MeshConnectivity& mesh,
                                              const std::vector<Eigen::Matrix2d>& abars,
                                              Eigen::VectorXd* deriv,
                                              std::vector<Eigen::Triplet<double>>* hessian,
                                              bool is_proj) override;
    double compute_magnitude_sq_change_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                      const MeshConnectivity& mesh,
                                                      const std::vector<Eigen::Matrix2d>& abars,
                                                      int face,
                                                      Eigen::VectorXd* deriv,
                                                      Eigen::MatrixXd* hessian,
                                                      bool is_proj) override;

    void test_compute_magnitude_sq_change_energy(const MeshConnectivity& mesh,
                                                 const std::vector<Eigen::Matrix2d>& abars,
                                                 const Eigen::VectorXd& edge_dofs) override;
    void test_compute_magnitude_sq_change_energy_perface(const MeshConnectivity& mesh,
                                                         const std::vector<Eigen::Matrix2d>& abars,
                                                         const Eigen::VectorXd& edge_dofs,
                                                         int face) override;

    // h^5 / 320 int_U ||Ibar^{-1}III||_SV^2 dA
    double compute_thirdFundamentalForm_energy(const Eigen::MatrixXd& cur_pos,
                                               const Eigen::VectorXd& edge_dofs,
                                               const MeshConnectivity& mesh,
                                               const std::vector<Eigen::Matrix2d>& abars,
                                               Eigen::VectorXd* deriv,
                                               std::vector<Eigen::Triplet<double>>* hessian,
                                               bool is_proj) override;
    double compute_thirdFundamentalForm_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                       const Eigen::VectorXd& edge_dofs,
                                                       const MeshConnectivity& mesh,
                                                       const std::vector<Eigen::Matrix2d>& abars,
                                                       int face,
                                                       Eigen::VectorXd* deriv,
                                                       Eigen::MatrixXd* hessian,
                                                       bool is_proj) override;
    void test_compute_thirdFundamentalForm_energy(const MeshConnectivity& mesh,
                                                  const std::vector<Eigen::Matrix2d>& abars,
                                                  const Eigen::MatrixXd& cur_pos,
                                                  const Eigen::VectorXd& edge_dofs) override;
    void test_compute_thirdFundamentalForm_energy_perface(const MeshConnectivity& mesh,
                                                          const std::vector<Eigen::Matrix2d>& abars,
                                                          const Eigen::MatrixXd& cur_pos,
                                                          const Eigen::VectorXd& edge_dofs,
                                                          int face) override;

    // TODO: Change this to private
public:
    MidedgeAngleGeneralTanFormulation m_sff;
};
}  // namespace LibShell
