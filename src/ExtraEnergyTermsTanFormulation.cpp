//
// Created by Zhen Chen on 12/8/24.
//
#include "../include/ExtraEnergyTermsTanFormulation.h"

#include "../include/MidedgeAngleGeneralTanFormulation.h"
#include "../include/types.h"

#include <iostream>
#include <tbb/parallel_for.h>

namespace LibShell {

double ExtraEnergyTermsTanFormulation::compute_magnitude_compression_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                      const MeshConnectivity& mesh,
                                                                      int face,
                                                                      Eigen::VectorXd* deriv,
                                                                      Eigen::MatrixXd* hessian,
                                                                      bool is_proj)  {
    double energy = 0;

    if (deriv) {
        deriv->setZero(3);
    }

    if (hessian) {
        hessian->setZero(3, 3);
    }


    return energy;
}

double ExtraEnergyTermsTanFormulation::compute_magnitude_compression_energy(const Eigen::VectorXd& edge_dofs,
                                                              const MeshConnectivity& mesh,
                                                              Eigen::VectorXd* deriv,
                                                              std::vector<Eigen::Triplet<double>>* hessian,
                                                              bool is_proj)  {
    double energy = 0;
    if (deriv) {
        deriv->setZero(edge_dofs.size());
    }
    if (hessian) {
        hessian->clear();
    }

    return energy;
}

double ExtraEnergyTermsTanFormulation::compute_vector_perp_tangent_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                                    const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::VectorXd* deriv,
                                                                    Eigen::MatrixXd* hessian,
                                                                    bool is_proj)  {
    std::vector<double> ni_b0(3);
    std::vector<double> ni_b1(3);

    constexpr int numExtraDOFs = MidedgeAngleGeneralTanFormulation::numExtraDOFs;

    std::vector<Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>> dni_b0(3);
    std::vector<Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>> dni_b1(3);

    std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>> hni_b0(3);
    std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>> hni_b1(3);

    for (int i = 0; i < 3; i++) {
        ni_b0[i] = m_sff.compute_nibj(mesh, cur_pos, edge_dofs, face, i, 0, deriv ? &dni_b0[i] : nullptr,
                                      hessian ? &hni_b0[i] : nullptr);
        ni_b1[i] = m_sff.compute_nibj(mesh, cur_pos, edge_dofs, face, i, 1, deriv ? &dni_b1[i] : nullptr,
                                      hessian ? &hni_b1[i] : nullptr);
    }

    double energy = 0;
    if (deriv) {
        deriv->setZero(18 + 3 * m_sff.numExtraDOFs);
    }

    if (hessian) {
        hessian->setZero(18 + 3 * m_sff.numExtraDOFs, 18 + 3 * m_sff.numExtraDOFs);
    }

    Eigen::Matrix2d abar_inv = abars[face].inverse();

    // exact formula, since it is a quadratic function of u,v
    // F = int_{0 <= u + v <= 1} (2u + 2v -1, 1-2u, 1-2v) K (2u + 2v -1, 1-2u, 1-2v)^T du dv * face_area * shear * thickness / 2
    //   = 1/6 (K(0, 0) + K(1, 1) + K(2, 2)) * face_area * shear * thickness / 2
    //   = 1/6 Tr(K) * face_area * shear * thickness / 2
    // K = N^Tdr Ibar_inv * dr^TN, N = [n0, n1, n2], dr = [b0, b1]
    // Let Ibar_inv = [a00, a01; a10, a11]
    // K_ij = a00 (b0^Tni) (b0^Tnj) + a11 (b1^Tni) (b1^Tnj) + a01 (b0^Tni) (b1^Tnj) + a10 (b1^Tni) (b0^Tnj)
    for(int i = 0; i < 3; i++) {
        double Kii = abar_inv(0, 0) * ni_b0[i] * ni_b0[i] + abar_inv(1, 1) * ni_b1[i] * ni_b1[i] + (abar_inv(0, 1) + abar_inv(1, 0)) * ni_b0[i] * ni_b1[i];
        energy += Kii / 6.0 * m_face_area[face] * m_shear * m_thickness / 2.0;
        if(deriv) {
            Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> dKii;
            dKii = 2 * abar_inv(0, 0) * dni_b0[i] * ni_b0[i] + 2 * abar_inv(1, 1) * dni_b1[i] * ni_b1[i] +  (abar_inv(0, 1) + abar_inv(1, 0))* (dni_b0[i] * ni_b1[i] + ni_b0[i] * dni_b1[i]);
            (*deriv) += dKii / 6.0 * m_face_area[face] * m_shear * m_thickness / 2.0;;
        }
        if(hessian) {
            Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> hKii;
            hKii = 2 * abar_inv(0, 0) * (hni_b0[i] * ni_b0[i] + dni_b0[i].transpose() * dni_b0[i]) + 2 * abar_inv(1, 1) * (hni_b1[i] * ni_b1[i] + dni_b1[i].transpose() * dni_b1[i]) + (abar_inv(0, 1) + abar_inv(1, 0)) * (hni_b0[i] * ni_b1[i] + hni_b1[i] * ni_b0[i] + dni_b0[i].transpose() * dni_b1[i] + dni_b1[i].transpose() * dni_b0[i]);
            (*hessian) += hKii / 6.0 * m_face_area[face] * m_shear * m_thickness / 2.0;
        }
    }

    if (hessian && is_proj) {
        proj_sym_matrix(*hessian, HessianProjectType::kMaxZero);
    }

    return energy;
}

double ExtraEnergyTermsTanFormulation::compute_vector_perp_tangent_energy(const Eigen::MatrixXd& cur_pos,
                                                            const Eigen::VectorXd& edge_dofs,
                                                            const MeshConnectivity& mesh,
                                                            const std::vector<Eigen::Matrix2d> &abars,
                                                            Eigen::VectorXd* deriv,
                                                            std::vector<Eigen::Triplet<double>>* hessian,
                                                            bool is_proj)  {
    double energy = 0;
    int nverts = cur_pos.rows();
    int nedges = mesh.nEdges();
    int nfaces = mesh.nFaces();

    Eigen::VectorXd extended_edge_dofs = Eigen::VectorXd::Zero(m_sff.numExtraDOFs * nedges);

    for(int i = 0; i < nedges; i++) {
        extended_edge_dofs(2 * i) = edge_dofs(i);
        extended_edge_dofs(2 * i + 1) = M_PI_2;
    }

    if (deriv) {
        deriv->setZero(3 * nverts + nedges);
    }

    if (hessian) {
        hessian->clear();
    }

    const int nedgedofs_extended = MidedgeAngleGeneralTanFormulation::numExtraDOFs;

    std::vector<double> energies(nfaces);
    std::vector<Eigen::VectorXd> face_derivs(nfaces);
    std::vector<Eigen::MatrixXd> face_hessians(nfaces);

    tbb::parallel_for(0, nfaces, [&](int i) {
        energies[i] =
            compute_vector_perp_tangent_energy_perface(cur_pos, extended_edge_dofs, mesh, abars, i, deriv ? &face_derivs[i] : nullptr, hessian ? &face_hessians[i] : nullptr, is_proj);
    });

    for (int i = 0; i < nfaces; i++) {
        Eigen::VectorXd& face_deriv = face_derivs[i];
        Eigen::MatrixXd& face_hess = face_hessians[i];

        energy += energies[i];

        if (deriv) {
            for (int j = 0; j < 3; j++) {
                deriv->segment<3>(3 * mesh.faceVertex(i, j)) += face_deriv.segment<3>(3 * j).transpose();
                int oppidx = mesh.vertexOppositeFaceEdge(i, j);
                if (oppidx != -1) {
                    deriv->segment<3>(3 * oppidx) += face_deriv.segment<3>(9 + 3 * j).transpose();
                }
                // the derivative w.r.t. the S2 angle is zero
                (*deriv)[3 * nverts + mesh.faceEdge(i, j)] += face_deriv(18 + nedgedofs_extended * j);
            }
        }
        if (hessian) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        for (int m = 0; m < 3; m++) {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                                      3 * mesh.faceVertex(i, k) + m,
                                                                      face_hess(3 * j + l, 3 * k + m)));
                            int oppidxk = mesh.vertexOppositeFaceEdge(i, k);
                            if (oppidxk != -1)
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * mesh.faceVertex(i, j) + l, 3 * oppidxk + m, face_hess(3 * j + l, 9 + 3 * k + m)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if (oppidxj != -1)
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * oppidxj + l, 3 * mesh.faceVertex(i, k) + m, face_hess(9 + 3 * j + l, 3 * k + m)));
                            if (oppidxj != -1 && oppidxk != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * oppidxk + m,
                                                                          face_hess(9 + 3 * j + l, 9 + 3 * k + m)));
                        }

                        // edge dofs
                        hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                                      3 * nverts + mesh.faceEdge(i, k),
                                                                      face_hess(3 * j + l, 18 + nedgedofs_extended * k)));
                        hessian->push_back(Eigen::Triplet<double>(3 * nverts + mesh.faceEdge(i, k),
                                                                  3 * mesh.faceVertex(i, j) + l,
                                                                  face_hess(18 + nedgedofs_extended * k, 3 * j + l)));
                        int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                        if (oppidxj != -1) {
                            hessian->push_back(Eigen::Triplet<double>(
                                3 * oppidxj + l, 3 * nverts + mesh.faceEdge(i, k),
                                face_hess(9 + 3 * j + l, 18 + nedgedofs_extended * k)));
                            hessian->push_back(Eigen::Triplet<double>(
                                3 * nverts + mesh.faceEdge(i, k), 3 * oppidxj + l,
                                face_hess(18 + nedgedofs_extended * k, 9 + 3 * j + l)));
                        }
                    }
                    hessian->push_back(
                                Eigen::Triplet<double>(3 * nverts + mesh.faceEdge(i, j),
                                                       3 * nverts + mesh.faceEdge(i, k),
                                                       face_hess(18 + nedgedofs_extended * j, 18 + nedgedofs_extended * k)));
                }
            }
        }
    }

    return energy;
}

double ExtraEnergyTermsTanFormulation::compute_magnitude_sq_change_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::VectorXd* deriv,
                                                                    Eigen::MatrixXd* hessian,
                                                                    bool is_proj)  {
    double energy = 0;
    if (deriv) {
        deriv->setZero(3);
    }

    if (hessian) {
        hessian->setZero(3, 3);
    }

    return energy;
}

double ExtraEnergyTermsTanFormulation::compute_magnitude_sq_change_energy(const Eigen::VectorXd& edge_dofs,
                                                            const MeshConnectivity& mesh,
                                                            const std::vector<Eigen::Matrix2d>& abars,
                                                            Eigen::VectorXd* deriv,
                                                            std::vector<Eigen::Triplet<double>>* hessian,
                                                            bool is_proj)  {
    double energy = 0;
    if (deriv) {
        deriv->setZero(edge_dofs.size());
    }
    if (hessian) {
        hessian->clear();
    }

    return energy;
}

void ExtraEnergyTermsTanFormulation::test_compute_magnitude_compression_energy_perface(const MeshConnectivity& mesh,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face)  {
}

void ExtraEnergyTermsTanFormulation::test_compute_magnitude_compression_energy(const MeshConnectivity& mesh,
                                                                 const Eigen::VectorXd& edge_dofs)  {
}

void ExtraEnergyTermsTanFormulation::test_compute_vector_perp_tangent_energy_perface(const MeshConnectivity& mesh,
const std::vector<Eigen::Matrix2d>& abars,
                                                                       const Eigen::MatrixXd& cur_pos,
                                                                       const Eigen::VectorXd& edge_dofs,
                                                                       int face)  {
    const int numExtraDOFs = MidedgeAngleGeneralTanFormulation::numExtraDOFs;
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(18 + 3);
        vars.setZero();
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = pos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if(opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = pos.row(opp_vid);
            }

            int eid = mesh.faceEdge(face, k);
            vars.segment<4>(18 + k * numExtraDOFs) = cur_edge_dofs.segment<4>(eid);
            vars(18 + k * numExtraDOFs + 1) = M_PI_2;
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }

            int eid = mesh.faceEdge(face, k);
            cur_edge_dofs(eid) = vars(18 + k * numExtraDOFs);
        }
    };

    Eigen::VectorXd vars = to_variables(cur_pos, edge_dofs);
    Eigen::MatrixXd pos = cur_pos;
    Eigen::VectorXd cur_edge_dofs = edge_dofs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, pos, cur_edge_dofs);
        Eigen::VectorXd dense_deriv;
        Eigen::MatrixXd dense_hess;
        double val = compute_vector_perp_tangent_energy_perface(pos, cur_edge_dofs, mesh, abars, face, deriv ? &dense_deriv : nullptr, hessian ? &dense_hess : nullptr, false);
        if(deriv) {
            *deriv = dense_deriv;
        }
        if(hessian) {
            std::vector<Eigen::Triplet<double>> T;
            for(int k = 0; k < dense_hess.rows(); k++) {
                for(int l = 0; l < dense_hess.cols(); l++) {
                    if(dense_hess(k, l) != 0) {
                        T.push_back(Eigen::Triplet<double>(k, l, dense_hess(k, l)));
                    }
                }
            }
            hessian->resize(18 + 3, 18 + 3);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}

void ExtraEnergyTermsTanFormulation::test_compute_vector_perp_tangent_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                               const Eigen::MatrixXd& cur_pos,
                                                               const Eigen::VectorXd& edge_dofs)  {
    int nedges = mesh.nEdges();
    int nverts = cur_pos.rows();
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(3 * nverts + nedges);
        vars.setZero();
        for(int i = 0; i < nverts; i++) {
            vars.segment<3>(3 * i) = pos.row(i);
        }
        for(int i = 0; i < nedges; i++) {
            vars(3 * nverts + i) = cur_edge_dofs(i);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        for(int i = 0; i < nverts; i++) {
            pos.row(i) = vars.segment<3>(3 * i);
        }
        for(int i = 0; i < nedges; i++) {
            cur_edge_dofs(i) = vars(3 * nverts + i);
        }
    };

    Eigen::VectorXd vars = to_variables(cur_pos, edge_dofs);
    Eigen::MatrixXd pos = cur_pos;
    Eigen::VectorXd cur_edge_dofs = edge_dofs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, pos, cur_edge_dofs);
        std::vector<Eigen::Triplet<double>> T;
        double val = compute_vector_perp_tangent_energy(pos, cur_edge_dofs, mesh, abars, deriv, hessian ? &T : nullptr, false);

        if(hessian) {
            hessian->resize( 3 * nverts + nedges,  3 * nverts + nedges);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}


void ExtraEnergyTermsTanFormulation::test_compute_magnitude_sq_change_energy_perface(const MeshConnectivity& mesh,
                                                                       const std::vector<Eigen::Matrix2d>& abars,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face)  {
}

void ExtraEnergyTermsTanFormulation::test_compute_magnitude_sq_change_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::VectorXd& edge_dofs)  {
}

double ExtraEnergyTermsTanFormulation::compute_thirdFundamentalForm_energy(
    const Eigen::MatrixXd& cur_pos,
    const Eigen::VectorXd& edge_dofs,
    const MeshConnectivity& mesh,
    const std::vector<Eigen::Matrix2d>& abars,
    Eigen::VectorXd* deriv,
    std::vector<Eigen::Triplet<double>>* hessian,
    bool is_proj) {
    double energy = 0;
    int nverts = cur_pos.rows();
    int nedges = mesh.nEdges();
    int nfaces = mesh.nFaces();

    if (deriv) {
        deriv->setZero(3 * nverts + nedges);
    }

    if (hessian) {
        hessian->clear();
    }

    return energy;
}

double ExtraEnergyTermsTanFormulation::compute_thirdFundamentalForm_energy_perface(
    const Eigen::MatrixXd& cur_pos,
    const Eigen::VectorXd& edge_dofs,
    const MeshConnectivity& mesh,
    const std::vector<Eigen::Matrix2d>& abars,
    int face,
    Eigen::VectorXd* derivative,
    Eigen::MatrixXd* hessian,
    bool is_proj) {
    // h^5 / 320 ||Ibar^{-1}III||_SV^2
    double dA = std::sqrt(abars[face].determinant()) / 2;
    double coeff = std::pow(m_thickness, 5) / 320.0;
    double energy = 0;
    if (derivative) {
        derivative->setZero(18 + 3);
    }
    if (hessian) {
        hessian->setZero(18 + 3, 18 + 3);
    }

    return energy;
}

void ExtraEnergyTermsTanFormulation::test_compute_thirdFundamentalForm_energy_perface(
    const MeshConnectivity& mesh,
    const std::vector<Eigen::Matrix2d>& abars,
    const Eigen::MatrixXd& cur_pos,
    const Eigen::VectorXd& edge_dofs,
    int face) {}


void ExtraEnergyTermsTanFormulation::test_compute_thirdFundamentalForm_energy(
    const MeshConnectivity& mesh,
    const std::vector<Eigen::Matrix2d>& abars,
    const Eigen::MatrixXd& cur_pos,
    const Eigen::VectorXd& edge_dofs) {}


}  // namespace LibShell