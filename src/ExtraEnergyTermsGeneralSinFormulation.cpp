//
// Created by Zhen Chen on 12/8/24.
//
#include "../include/ExtraEnergyTermsGeneralSinFormulation.h"
#include "../include/types.h"

#include <iostream>
#include <tbb/parallel_for.h>

namespace LibShell {

double ExtraEnergyTermsGeneralSinFormulation::compute_magnitude_compression_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                      const MeshConnectivity& mesh,
                                                                      int face,
                                                                      Eigen::VectorXd* deriv,
                                                                      Eigen::MatrixXd* hessian,
                                                                      bool is_proj) {
    double energy = 0;

    if (deriv) {
        deriv->setZero(3);
    }

    if (hessian) {
        hessian->setZero(3, 3);
    }


    return energy;
}

double ExtraEnergyTermsGeneralSinFormulation::compute_magnitude_compression_energy(const Eigen::VectorXd& edge_dofs,
                                                              const MeshConnectivity& mesh,
                                                              Eigen::VectorXd* deriv,
                                                              std::vector<Eigen::Triplet<double>>* hessian,
                                                              bool is_proj) {
    double energy = 0;
    if (deriv) {
        deriv->setZero(edge_dofs.size());
    }
    if (hessian) {
        hessian->clear();
    }

    return energy;
}

double ExtraEnergyTermsGeneralSinFormulation::compute_vector_perp_tangent_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                                    const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::VectorXd* deriv,
                                                                    Eigen::MatrixXd* hessian,
                                                                    bool is_proj)  {
    std::vector<double> ni_b0(3);
    std::vector<double> ni_b1(3);

    constexpr int numExtraDOFs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;

    assert(numExtraDOFs == m_sff.numExtraDOFs);

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

double ExtraEnergyTermsGeneralSinFormulation::compute_vector_perp_tangent_energy(const Eigen::MatrixXd& cur_pos,
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

    if (deriv) {
        deriv->setZero(3 * nverts + m_sff.numExtraDOFs * nedges);
    }

    if (hessian) {
        hessian->clear();
    }

    constexpr int nedgedofs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;

    std::vector<double> energies(nfaces);
    std::vector<Eigen::VectorXd> face_derivs(nfaces);
    std::vector<Eigen::MatrixXd> face_hessians(nfaces);

    tbb::parallel_for(0, nfaces, [&](int i) {
        energies[i] =
            compute_vector_perp_tangent_energy_perface(cur_pos, edge_dofs, mesh, abars, i, deriv ? &face_derivs[i] : nullptr, hessian ? &face_hessians[i] : nullptr, is_proj);
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
                for (int k = 0; k < nedgedofs; k++) {
                    (*deriv)[3 * nverts + nedgedofs * mesh.faceEdge(i, j) + k] += face_deriv(18 + nedgedofs * j + k);
                }
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
                        for (int m = 0; m < nedgedofs; m++) {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                                      3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m,
                                                                      face_hess(3 * j + l, 18 + nedgedofs * k + m)));
                            hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m,
                                                                      3 * mesh.faceVertex(i, j) + l,
                                                                      face_hess(18 + nedgedofs * k + m, 3 * j + l)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if (oppidxj != -1) {
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * oppidxj + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m,
                                    face_hess(9 + 3 * j + l, 18 + nedgedofs * k + m)));
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * oppidxj + l,
                                    face_hess(18 + nedgedofs * k + m, 9 + 3 * j + l)));
                            }
                        }
                    }
                    for (int m = 0; m < nedgedofs; m++) {
                        for (int n = 0; n < nedgedofs; n++) {
                            hessian->push_back(
                                Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, j) + m,
                                                       3 * nverts + nedgedofs * mesh.faceEdge(i, k) + n,
                                                       face_hess(18 + nedgedofs * j + m, 18 + nedgedofs * k + n)));
                        }
                    }
                }
            }
        }
    }

    return energy;
}

double ExtraEnergyTermsGeneralSinFormulation::compute_magnitude_sq_change_energy_perface(const Eigen::VectorXd& edge_dofs,
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

double ExtraEnergyTermsGeneralSinFormulation::compute_magnitude_sq_change_energy(const Eigen::VectorXd& edge_dofs,
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

void ExtraEnergyTermsGeneralSinFormulation::test_compute_magnitude_compression_energy_perface(const MeshConnectivity& mesh,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face)  {
}

void ExtraEnergyTermsGeneralSinFormulation::test_compute_magnitude_compression_energy(const MeshConnectivity& mesh,
                                                                 const Eigen::VectorXd& edge_dofs)  {
}

void ExtraEnergyTermsGeneralSinFormulation::test_compute_vector_perp_tangent_energy_perface(const MeshConnectivity& mesh,
const std::vector<Eigen::Matrix2d>& abars,
                                                                       const Eigen::MatrixXd& cur_pos,
                                                                       const Eigen::VectorXd& edge_dofs,
                                                                       int face)  {
    static constexpr int numExtraDOFs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = pos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if(opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = pos.row(opp_vid);
            }

            int eid = mesh.faceEdge(face, k);
            vars.segment<numExtraDOFs>(18 + numExtraDOFs * k) = cur_edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }

            int eid = mesh.faceEdge(face, k);
            cur_edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * k);
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
            hessian->resize(18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}

void ExtraEnergyTermsGeneralSinFormulation::test_compute_vector_perp_tangent_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                               const Eigen::MatrixXd& cur_pos,
                                                               const Eigen::VectorXd& edge_dofs)  {
    int nedges = mesh.nEdges();
    int nverts = cur_pos.rows();
    static constexpr int numExtraDOFs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(3 * nverts + numExtraDOFs * nedges);
        vars.setZero();
        for(int i = 0; i < nverts; i++) {
            vars.segment<3>(3 * i) = pos.row(i);
        }
        for(int i = 0; i < nedges; i++) {
            vars.segment<numExtraDOFs>(3 * nverts + numExtraDOFs * i) = cur_edge_dofs.segment<numExtraDOFs>(numExtraDOFs * i);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        assert(vars.size() == 3 * nverts + numExtraDOFs * nedges);
        for(int i = 0; i < nverts; i++) {
            pos.row(i) = vars.segment<3>(3 * i);
        }
        for(int i = 0; i < nedges; i++) {
            cur_edge_dofs.segment<numExtraDOFs>(numExtraDOFs * i) = vars.segment<numExtraDOFs>(3 * nverts + numExtraDOFs * i);
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
            hessian->resize( 3 * nverts + numExtraDOFs * nedges,  3 * nverts + numExtraDOFs * nedges);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}


void ExtraEnergyTermsGeneralSinFormulation::test_compute_magnitude_sq_change_energy_perface(const MeshConnectivity& mesh,
                                                                       const std::vector<Eigen::Matrix2d>& abars,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face)  {
}

void ExtraEnergyTermsGeneralSinFormulation::test_compute_magnitude_sq_change_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::VectorXd& edge_dofs)  {
}

double ExtraEnergyTermsGeneralSinFormulation::compute_thirdFundamentalForm_energy(
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
        deriv->setZero(3 * nverts + m_sff.numExtraDOFs * nedges);
    }

    if (hessian) {
        hessian->clear();
    }

    constexpr int nedgedofs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;

    std::vector<double> energies(nfaces);
    std::vector<Eigen::VectorXd> face_derivs(nfaces);
    std::vector<Eigen::MatrixXd> face_hessians(nfaces);

    tbb::parallel_for(0, nfaces, [&](int i) {
        // for(int i = 0; i < nfaces; i++) {
        energies[i] =
            compute_thirdFundamentalForm_energy_perface(cur_pos, edge_dofs, mesh, abars, i, deriv ? &face_derivs[i] : nullptr, hessian ? &face_hessians[i] : nullptr, is_proj);
    }
    );

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
                for (int k = 0; k < nedgedofs; k++) {
                    (*deriv)[3 * nverts + nedgedofs * mesh.faceEdge(i, j) + k] += face_deriv(18 + nedgedofs * j + k);
                }
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
                        for (int m = 0; m < nedgedofs; m++) {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                                      3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m,
                                                                      face_hess(3 * j + l, 18 + nedgedofs * k + m)));
                            hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m,
                                                                      3 * mesh.faceVertex(i, j) + l,
                                                                      face_hess(18 + nedgedofs * k + m, 3 * j + l)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if (oppidxj != -1) {
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * oppidxj + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m,
                                    face_hess(9 + 3 * j + l, 18 + nedgedofs * k + m)));
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * oppidxj + l,
                                    face_hess(18 + nedgedofs * k + m, 9 + 3 * j + l)));
                            }
                        }
                    }
                    for (int m = 0; m < nedgedofs; m++) {
                        for (int n = 0; n < nedgedofs; n++) {
                            hessian->push_back(
                                Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, j) + m,
                                                       3 * nverts + nedgedofs * mesh.faceEdge(i, k) + n,
                                                       face_hess(18 + nedgedofs * j + m, 18 + nedgedofs * k + n)));
                        }
                    }
                }
            }
        }
    }

    return energy;
}

double ExtraEnergyTermsGeneralSinFormulation::compute_thirdFundamentalForm_energy_perface(
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
        derivative->setZero(18 + 3 * m_sff.numExtraDOFs);
    }
    if (hessian) {
        hessian->setZero(18 + 3 * m_sff.numExtraDOFs, 18 + 3 * m_sff.numExtraDOFs);
    }

    constexpr int nedgedofs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;

    Eigen::Matrix<double, 4, 18 + 3 * nedgedofs> III_deriv;
    std::vector<Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs>> III_hess;
    Eigen::Matrix2d III = m_sff.thirdFundamentalForm(mesh, cur_pos, edge_dofs, face, (derivative || hessian) ? &III_deriv : nullptr,
                                                    (derivative || hessian) ? &III_hess : nullptr);
    Eigen::Matrix2d abarinv = abars[face].inverse();
    Eigen::Matrix2d M = abarinv * (III);

    double lameAlpha = m_youngs * m_poisson / (1.0 - m_poisson * m_poisson);
    double lameBeta = m_youngs / 2.0 / (1.0 + m_poisson);


    double StVK = 0.5 * lameAlpha * pow(M.trace(), 2) + lameBeta * (M * M).trace();
    double result = coeff * dA * StVK;

    if (derivative) {
        Eigen::Matrix2d temp = lameAlpha * M.trace() * abarinv + 2 * lameBeta * M * abarinv;
        *derivative = coeff * dA * III_deriv.transpose() * Eigen::Map<Eigen::Vector4d>(temp.data());
    }

    if (hessian) {
        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> inner = III_deriv.transpose() * Eigen::Map<Eigen::Vector4d>(abarinv.data());
        *hessian = lameAlpha * inner.transpose() * inner;

        Eigen::Matrix2d Mainv = M * abarinv;
        for (int i = 0; i < 4; ++i)  // iterate over Mainv and abarinv as if they were vectors
            *hessian += (lameAlpha * M.trace() * abarinv(i) + 2 * lameBeta * Mainv(i)) * III_hess[i];

        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> inner00 = abarinv(0, 0) * III_deriv.row(0) + abarinv(0, 1) * III_deriv.row(2);
        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> inner01 = abarinv(0, 0) * III_deriv.row(1) + abarinv(0, 1) * III_deriv.row(3);
        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> inner10 = abarinv(1, 0) * III_deriv.row(0) + abarinv(1, 1) * III_deriv.row(2);
        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> inner11 = abarinv(1, 0) * III_deriv.row(1) + abarinv(1, 1) * III_deriv.row(3);
        *hessian += 2 * lameBeta * inner00.transpose() * inner00;
        *hessian += 2 * lameBeta * (inner01.transpose() * inner10 + inner10.transpose() * inner01);
        *hessian += 2 * lameBeta * inner11.transpose() * inner11;

        *hessian *= coeff * dA;

        if(is_proj) {
            proj_sym_matrix(*hessian, HessianProjectType::kMaxZero);
        }
    }

    return result;
}

void ExtraEnergyTermsGeneralSinFormulation::test_compute_thirdFundamentalForm_energy_perface(
    const MeshConnectivity& mesh,
    const std::vector<Eigen::Matrix2d>& abars,
    const Eigen::MatrixXd& cur_pos,
    const Eigen::VectorXd& edge_dofs,
    int face) {
    static constexpr int numExtraDOFs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = pos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if(opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = pos.row(opp_vid);
            }

            int eid = mesh.faceEdge(face, k);
            vars.segment<numExtraDOFs>(18 + numExtraDOFs * k) = cur_edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }

            int eid = mesh.faceEdge(face, k);
            cur_edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * k);
        }
    };

    Eigen::VectorXd vars = to_variables(cur_pos, edge_dofs);
    Eigen::MatrixXd pos = cur_pos;
    Eigen::VectorXd cur_edge_dofs = edge_dofs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, pos, cur_edge_dofs);
        Eigen::VectorXd dense_deriv;
        Eigen::MatrixXd dense_hess;
        double val = compute_thirdFundamentalForm_energy_perface(pos, cur_edge_dofs, mesh, abars, face, deriv ? &dense_deriv : nullptr, hessian ? &dense_hess : nullptr, false);
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
            hessian->resize(18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}


void ExtraEnergyTermsGeneralSinFormulation::test_compute_thirdFundamentalForm_energy(
    const MeshConnectivity& mesh,
    const std::vector<Eigen::Matrix2d>& abars,
    const Eigen::MatrixXd& cur_pos,
    const Eigen::VectorXd& edge_dofs) {
    int nedges = mesh.nEdges();
    int nverts = cur_pos.rows();
    static constexpr int numExtraDOFs = MidedgeAngleGeneralSinFormulation::numExtraDOFs;
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(3 * nverts + numExtraDOFs * nedges);
        vars.setZero();
        for(int i = 0; i < nverts; i++) {
            vars.segment<3>(3 * i) = pos.row(i);
        }
        for(int i = 0; i < nedges; i++) {
            vars.segment<numExtraDOFs>(3 * nverts + numExtraDOFs * i) = cur_edge_dofs.segment<numExtraDOFs>(numExtraDOFs * i);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        assert(vars.size() == 3 * nverts + numExtraDOFs * nedges);
        for(int i = 0; i < nverts; i++) {
            pos.row(i) = vars.segment<3>(3 * i);
        }
        for(int i = 0; i < nedges; i++) {
            cur_edge_dofs.segment<numExtraDOFs>(numExtraDOFs * i) = vars.segment<numExtraDOFs>(3 * nverts + numExtraDOFs * i);
        }
    };

    Eigen::VectorXd vars = to_variables(cur_pos, edge_dofs);
    Eigen::MatrixXd pos = cur_pos;
    Eigen::VectorXd cur_edge_dofs = edge_dofs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, pos, cur_edge_dofs);
        std::vector<Eigen::Triplet<double>> T;
        double val = compute_thirdFundamentalForm_energy(pos, cur_edge_dofs, mesh, abars, deriv, hessian ? &T : nullptr, false);

        if(hessian) {
            hessian->resize( 3 * nverts + numExtraDOFs * nedges,  3 * nverts + numExtraDOFs * nedges);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}


}  // namespace LibShell