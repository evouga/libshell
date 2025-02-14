//
// Created by Zhen Chen on 12/8/24.
//
#include "../include/ExtraEnergyTermsGeneralFormulation.h"
#include "../include/types.h"

#include <iostream>
#include <tbb/parallel_for.h>

namespace LibShell {

double ExtraEnergyTermsGeneralFormulation::compute_magnitude_compression_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                      const MeshConnectivity& mesh,
                                                                      int face,
                                                                      Eigen::VectorXd* deriv,
                                                                      Eigen::MatrixXd* hessian,
                                                                      bool is_proj)  {
    std::vector<double> m(3);

    for (int i = 0; i < 3; i++) {
        int eid = mesh.faceEdge(face, i);
        int efid = mesh.faceEdgeOrientation(face, i);
        double mi0 = edge_dofs(eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid);
        m[i] = mi0;
    }

    if (deriv) {
        deriv->setZero(3);
    }

    if (hessian) {
        hessian->setZero(3, 3);
    }

    // m = (2u + 2v - 1) m0 + (1 - 2u) m1 + (1 - 2v) m2
    // int_u (m^2 - 1)^2 = 1/30 (15 + 3 m0^4 + 3 m1^4 - 10 m2^2 + 3 m2^4 - 8 m0 m1 m2 (m1 + m2) + 10 m1^2 (-1 + m2^2) + 2 m0^2 (-5 + 5 m1^2 - 4 m1 m2 + 5 m2^2))
    double energy = 1.0 / 30.0 * (15.0 + 3 * m[0] * m[0] * m[0] * m[0] + 3 * m[1] * m[1] * m[1] * m[1] - 10 * m[2] * m[2] +
                                  3 * m[2] * m[2] * m[2] * m[2] - 8 * m[0] * m[1] * m[2] * (m[1] + m[2]) +
                                  10 * m[1] * m[1] * (-1 + m[2] * m[2]) +
                                  2 * m[0] * m[0] * (-5 + 5 * m[1] * m[1] - 4 * m[1] * m[2] + 5 * m[2] * m[2]));
    energy *= m_face_area[face] * m_youngs * m_thickness / 4;

    if(deriv) {
        // 1/15 (6 m0^3 - 4 m1 m2 (m1 + m2) + 2 m0 (-5 + 5 m1^2 - 4 m1 m2 + 5 m2^2))
        (*deriv)[0] = 1.0 / 15.0 * (6 * m[0] * m[0] * m[0] - 4 * m[1] * m[2] * (m[1] + m[2]) +
                                    2 * m[0] * (-5 + 5 * m[1] * m[1] - 4 * m[1] * m[2] + 5 * m[2] * m[2]));

        // 2/15 (3 m1^3 - 2 m0 m2 (m0 + m2) + m1 (-5 + 5 m0^2 - 4 m0 m2 + 5 m2^2))
        (*deriv)[1] = 2.0 / 15.0 * (3 * m[1] * m[1] * m[1] - 2 * m[0] * m[2] * (m[0] + m[2]) +
                                    m[1] * (-5 + 5 * m[0] * m[0] - 4 * m[0] * m[2] + 5 * m[2] * m[2]));

        // 1/15 (-4 m0 m1 (m0 + m1) + 2 (-5 + 5 m0^2 - 4 m0 m1 + 5 m1^2) m2 + 6 m2^3)
        (*deriv)[2] = 1.0 / 15.0 * (-4 * m[0] * m[1] * (m[0] + m[1]) +
                                    2 * (-5 + 5 * m[0] * m[0] - 4 * m[0] * m[1] + 5 * m[1] * m[1]) * m[2] +
                                    6 * m[2] * m[2] * m[2]);

        (*deriv) *= m_face_area[face] * m_youngs * m_thickness / 4;
    }

    if(hessian) {
        // H(0, 0) = 2/15 (-5 + 9 m0^2 + 5 m1^2 - 4 m1 m2 + 5 m2^2)
        (*hessian)(0, 0) = 2.0 / 15.0 * (-5 + 9 * m[0] * m[0] + 5 * m[1] * m[1] - 4 * m[1] * m[2] + 5 * m[2] * m[2]);

        // H(0, 1) = 1/15 (20 m0 m1 - 8 (m0 + m1) m2 - 4 m2^2)
        (*hessian)(0, 1) = 1.0 / 15.0 * (20 * m[0] * m[1] - 8 * (m[0] + m[1]) * m[2] - 4 * m[2] * m[2]);

        // H(0, 2) = -(4/15) (2 m0 m1 + m1^2 - 5 m0 m2 + 2 m1 m2)
        (*hessian)(0, 2) = -(4.0 / 15.0) * (2 * m[0] * m[1] + m[1] * m[1] - 5 * m[0] * m[2] + 2 * m[1] * m[2]);

        // H(1, 0) = H(0, 1)
        (*hessian)(1, 0) = (*hessian)(0, 1);

        // H(1, 1) = 2/15 (-5 + 5 m0^2 + 9 m1^2 - 4 m0 m2 + 5 m2^2)
        (*hessian)(1, 1) = 2.0 / 15.0 * (-5 + 5 * m[0] * m[0] + 9 * m[1] * m[1] - 4 * m[0] * m[2] + 5 * m[2] * m[2]);

        // H(1, 2) = -(4/15) (m0^2 - 5 m1 m2 + 2 m0 (m1 + m2))
        (*hessian)(1, 2) = -(4.0 / 15.0) * (m[0] * m[0] - 5 * m[1] * m[2] + 2 * m[0] * (m[1] + m[2]));

        // H(2, 0) = H(0, 2)
        (*hessian)(2, 0) = (*hessian)(0, 2);

        // H(2, 1) = H(1, 2)
        (*hessian)(2, 1) = (*hessian)(1, 2);

        // H(2, 2) = 2/15 (-5 + 5 m0^2 - 4 m0 m1 + 5 m1^2 + 9 m2^2)
        (*hessian)(2, 2) = 2.0 / 15.0 * (-5 + 5 * m[0] * m[0] - 4 * m[0] * m[1] + 5 * m[1] * m[1] + 9 * m[2] * m[2]);

        if(is_proj) {
            proj_sym_matrix(*hessian, HessianProjectType::kMaxZero);
        }

        (*hessian) *= m_face_area[face] * m_youngs * m_thickness / 4;
    }

    return energy;
}

double ExtraEnergyTermsGeneralFormulation::compute_magnitude_compression_energy(const Eigen::VectorXd& edge_dofs,
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

    int nfaces = mesh.nFaces();

    std::vector<double> energies(nfaces);
    std::vector<Eigen::VectorXd> face_derivs(nfaces);
    std::vector<Eigen::MatrixXd> face_hessians(nfaces);

    tbb::parallel_for(0, nfaces, [&](int i) {
        energies[i] =
            compute_magnitude_compression_energy_perface(
             edge_dofs, mesh, i, deriv ? &face_derivs[i] : nullptr, hessian ? &face_hessians[i] : nullptr, is_proj);
    });

    for (int fid = 0; fid < nfaces; fid++) {
        Eigen::VectorXd& face_deriv = face_derivs[fid];
        Eigen::MatrixXd& face_hessian = face_hessians[fid];
        double face_energy = energies[fid];
        energy += face_energy;
        if (deriv) {
            for (int j = 0; j < 3; j++) {
                int eid = mesh.faceEdge(fid, j);
                int efid = mesh.faceEdgeOrientation(fid, j);
                int id = eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid;
                (*deriv)[id] += face_deriv(j);
            }
        }
        if (hessian) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int eid0 = mesh.faceEdge(fid, j);
                    int efid0 = mesh.faceEdgeOrientation(fid, j);
                    int id0 = eid0 * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid0;

                    int eid1 = mesh.faceEdge(fid, k);
                    int efid1 = mesh.faceEdgeOrientation(fid, k);
                    int id1 = eid1 * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid1;

                    hessian->push_back(Eigen::Triplet<double>(id0, id1, face_hessian(j, k)));
                }
            }
        }
    }

    return energy;
}

double ExtraEnergyTermsGeneralFormulation::compute_vector_perp_tangent_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                                    const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::VectorXd* deriv,
                                                                    Eigen::MatrixXd* hessian,
                                                                    bool is_proj)  {
    std::vector<double> ni_b0(3);
    std::vector<double> ni_b1(3);

    constexpr int numExtraDOFs = MidedgeAngleGeneralFormulation::numExtraDOFs;

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

double ExtraEnergyTermsGeneralFormulation::compute_vector_perp_tangent_energy(const Eigen::MatrixXd& cur_pos,
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

    constexpr int nedgedofs = MidedgeAngleGeneralFormulation::numExtraDOFs;

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

double ExtraEnergyTermsGeneralFormulation::compute_magnitude_sq_change_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::VectorXd* deriv,
                                                                    Eigen::MatrixXd* hessian,
                                                                    bool is_proj)  {
    double sum = 0;
    std::vector<double> m(3);

    for (int i = 0; i < 3; i++) {
        int eid = mesh.faceEdge(face, i);
        int efid = mesh.faceEdgeOrientation(face, i);
        double mi0 = edge_dofs(eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid);
        m[i] = mi0;
    }

    if (deriv) {
        deriv->setZero(3);
    }

    if (hessian) {
        hessian->setZero(3, 3);
    }

    Eigen::Matrix2d abar_inv = abars[face].inverse();
    // abar_inv = abar_inv.transpose() * abar_inv;

    for (int i = 0; i < m_quad_points.size(); i++) {
        double u = m_quad_points[i].u;
        double v = m_quad_points[i].v;
        double w = m_quad_points[i].weight * m_face_area[face] * m_shear * m_thickness * m_thickness * m_thickness / 24;

        double mag = (2 * u + 2 * v - 1) * m[0] + (1 - 2 * u) * m[1] + (1 - 2 * v) * m[2];
        double mag_sq = mag * mag;
        Eigen::Vector2d dm;
        dm << 2 * (m[0] - m[1]), 2 * (m[0] - m[2]);
        double dm_sq = dm.dot(abar_inv * dm);
        sum += w * mag_sq * dm_sq;

        if (deriv || hessian) {
            Eigen::Vector3d mag_sq_deriv, mag_deriv;
            mag_deriv << 2 * u + 2 * v - 1, 1 - 2 * u, 1 - 2 * v;
            mag_sq_deriv = 2 * mag * mag_deriv;

            Eigen::Matrix<double, 2, 3> dm_deriv;
            dm_deriv << 2, -2, 0, 2, 0, -2;

            Eigen::Vector3d dm_sq_deriv;
            dm_sq_deriv = 2 * dm_deriv.transpose() * (abar_inv * dm);

            if (deriv) {
                (*deriv) += w * mag_sq * dm_sq_deriv + w * dm_sq * mag_sq_deriv;
            }

            if (hessian) {
                (*hessian) += w * mag_sq_deriv * dm_sq_deriv.transpose() + w * dm_sq_deriv * mag_sq_deriv.transpose();

                Eigen::Matrix3d mag_sq_hess = 2 * mag_deriv * mag_deriv.transpose();

                Eigen::Matrix3d dm_sq_hess = 2 * dm_deriv.transpose() * abar_inv * dm_deriv;

                (*hessian) += w * mag_sq_hess * dm_sq + w * mag_sq * dm_sq_hess;

            }
        }
    }

    if (hessian && is_proj) {
        proj_sym_matrix(*hessian, HessianProjectType::kMaxZero);
    }
    return sum;
}

double ExtraEnergyTermsGeneralFormulation::compute_magnitude_sq_change_energy(const Eigen::VectorXd& edge_dofs,
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

    std::vector<double> energies(mesh.nFaces());
    std::vector<Eigen::VectorXd> face_derivs(mesh.nFaces());
    std::vector<Eigen::MatrixXd> face_hessians(mesh.nFaces());

    tbb::parallel_for(0, (int)(mesh.nFaces()), [&](int i) {
        energies[i] =
            compute_magnitude_sq_change_energy_perface(
            edge_dofs, mesh, abars, i, deriv ? &face_derivs[i] : nullptr, hessian ? &face_hessians[i] : nullptr, is_proj);
    });

    int nfaces = mesh.nFaces();
    for (int fid = 0; fid < nfaces; fid++) {

        double face_energy = energies[fid];

        Eigen::VectorXd &face_deriv = face_derivs[fid];
        Eigen::MatrixXd &face_hessian = face_hessians[fid];

        energy += face_energy;
        if (deriv) {
            for (int j = 0; j < 3; j++) {
                int eid = mesh.faceEdge(fid, j);
                int efid = mesh.faceEdgeOrientation(fid, j);
                int id = eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid;
                (*deriv)[id] += face_deriv(j);
            }
        }
        if (hessian) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int eid0 = mesh.faceEdge(fid, j);
                    int efid0 = mesh.faceEdgeOrientation(fid, j);
                    int id0 = eid0 * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid0;

                    int eid1 = mesh.faceEdge(fid, k);
                    int efid1 = mesh.faceEdgeOrientation(fid, k);
                    int id1 = eid1 * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid1;

                    hessian->push_back(Eigen::Triplet<double>(id0, id1, face_hessian(j, k)));
                }
            }
        }
    }

    return energy;
}

void ExtraEnergyTermsGeneralFormulation::test_compute_magnitude_compression_energy_perface(const MeshConnectivity& mesh,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face)  {
    auto to_variables = [&](const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(3);
        vars.setZero();
        for (int i = 0; i < 3; i++) {
            int eid = mesh.faceEdge(face, i);
            int efid = mesh.faceEdgeOrientation(face, i);
            double mi0 = cur_edge_dofs(eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid);
            vars[i] = mi0;
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::VectorXd& cur_edge_dofs) {
      for(int i = 0; i < 3; i++) {
          int eid = mesh.faceEdge(face, i);
          int efid = mesh.faceEdgeOrientation(face, i);
          int id = eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid;
          cur_edge_dofs[id] = vars[i];
      }
    };


    Eigen::VectorXd cur_edge_dofs = edge_dofs;
    Eigen::Vector3d vars = to_variables(cur_edge_dofs);

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, cur_edge_dofs);
        Eigen::VectorXd dense_deriv;
        Eigen::MatrixXd dense_hess;
        double val = compute_magnitude_compression_energy_perface(
            cur_edge_dofs, mesh, face, deriv ? &dense_deriv : nullptr, hessian ? &dense_hess : nullptr, false);
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
            hessian->resize(3, 3);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}

void ExtraEnergyTermsGeneralFormulation::test_compute_magnitude_compression_energy(const MeshConnectivity& mesh,
                                                                 const Eigen::VectorXd& edge_dofs)  {
    auto to_variables = [&](const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars = cur_edge_dofs;
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::VectorXd& cur_edge_dofs) {
        cur_edge_dofs = vars;
    };

    Eigen::VectorXd cur_edge_dofs = edge_dofs;
    Eigen::VectorXd vars = to_variables(cur_edge_dofs);

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, cur_edge_dofs);
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double val = compute_magnitude_compression_energy(
            cur_edge_dofs, mesh, deriv, hessian ? &hessian_triplets : nullptr, false);

        if(hessian) {
            hessian->resize(x.size(), x.size());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}

void ExtraEnergyTermsGeneralFormulation::test_compute_vector_perp_tangent_energy_perface(const MeshConnectivity& mesh,
const std::vector<Eigen::Matrix2d>& abars,
                                                                       const Eigen::MatrixXd& cur_pos,
                                                                       const Eigen::VectorXd& edge_dofs,
                                                                       int face)  {
    constexpr int numExtraDOFs = MidedgeAngleGeneralFormulation::numExtraDOFs;
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
            vars.segment<4>(18 + numExtraDOFs * k) = cur_edge_dofs.segment<4>(eid * numExtraDOFs);
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
            cur_edge_dofs.segment<4>(eid * numExtraDOFs) = vars.segment<4>(18 + numExtraDOFs * k);
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

void ExtraEnergyTermsGeneralFormulation::test_compute_vector_perp_tangent_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                               const Eigen::MatrixXd& cur_pos,
                                                               const Eigen::VectorXd& edge_dofs) {
    int nedges = mesh.nEdges();
    int nverts = cur_pos.rows();
    constexpr int numExtraDOFs = MidedgeAngleGeneralFormulation::numExtraDOFs;
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


void ExtraEnergyTermsGeneralFormulation::test_compute_magnitude_sq_change_energy_perface(const MeshConnectivity& mesh,
                                                                       const std::vector<Eigen::Matrix2d>& abars,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face)  {
    auto to_variables = [&](const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(3);
        vars.setZero();
        for (int i = 0; i < 3; i++) {
            int eid = mesh.faceEdge(face, i);
            int efid = mesh.faceEdgeOrientation(face, i);
            double mi0 = cur_edge_dofs(eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid);
            vars[i] = mi0;
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::VectorXd& cur_edge_dofs) {
      for(int i = 0; i < 3; i++) {
          int eid = mesh.faceEdge(face, i);
          int efid = mesh.faceEdgeOrientation(face, i);
          int id = eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid;
          cur_edge_dofs[id] = vars[i];
      }
    };


    Eigen::VectorXd cur_edge_dofs = edge_dofs;
    Eigen::Vector3d vars = to_variables(cur_edge_dofs);

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, cur_edge_dofs);
        Eigen::VectorXd dense_deriv;
        Eigen::MatrixXd dense_hess;
        double val = compute_magnitude_sq_change_energy_perface(
            cur_edge_dofs, mesh, abars, face, deriv ? &dense_deriv : nullptr, hessian ? &dense_hess : nullptr, false);
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
            hessian->resize(3, 3);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}

void ExtraEnergyTermsGeneralFormulation::test_compute_magnitude_sq_change_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::VectorXd& edge_dofs)  {
    auto to_variables = [&](const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars = cur_edge_dofs;
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::VectorXd& cur_edge_dofs) {
        cur_edge_dofs = vars;
    };

    Eigen::VectorXd cur_edge_dofs = edge_dofs;
    Eigen::VectorXd vars = to_variables(cur_edge_dofs);

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, cur_edge_dofs);
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double val = compute_magnitude_sq_change_energy(
            cur_edge_dofs, mesh, abars, deriv, hessian ? &hessian_triplets : nullptr, false);

        if(hessian) {
            hessian->resize(x.size(), x.size());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
        }
        return val;
    };

    TestFuncGradHessian(func, vars);
}

}  // namespace LibShell