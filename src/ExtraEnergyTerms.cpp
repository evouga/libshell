//
// Created by Zhen Chen on 12/8/24.
//
#include "../include/ExtraEnergyTerms.h"
#include "../include/types.h"

#include <iostream>

static void TestFuncGradHessian(
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

namespace LibShell {

template <class DerivedA>
static void proj_sym_matrix(Eigen::MatrixBase<DerivedA>& A, const HessianProjectType& projType) {
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

void ExtraEnergyTerms::initialization(const Eigen::MatrixXd& rest_pos,
                                      const MeshConnectivity& mesh,
                                      double youngs,
                                      double shear,
                                      double thickness,
                                      int quad_oder) {
    m_youngs = youngs;
    m_shear = shear;
    m_thickness = thickness;
    m_quad_points = build_quadrature_points(quad_oder);
    m_sff.initializeEdgeFaceBasisSign(mesh, rest_pos);

    m_face_area.resize(mesh.nFaces());
    for (int f = 0; f < mesh.nFaces(); f++) {
        Eigen::Vector3d e1 = rest_pos.row(mesh.faceVertex(f, 1)) - rest_pos.row(mesh.faceVertex(f, 0));
        Eigen::Vector3d e2 = rest_pos.row(mesh.faceVertex(f, 2)) - rest_pos.row(mesh.faceVertex(f, 0));
        m_face_area[f] = 0.5 * e1.cross(e2).norm();
    }
}

double ExtraEnergyTerms::compute_magnitude_compression_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                      const MeshConnectivity& mesh,
                                                                      int face,
                                                                      Eigen::Vector3d* deriv,
                                                                      Eigen::Matrix<double, 3, 3>* hessian,
                                                                      bool is_proj) {
    double sum = 0;
    std::vector<double> m(3);

    for (int i = 0; i < 3; i++) {
        int eid = mesh.faceEdge(face, i);
        int efid = mesh.faceEdgeOrientation(face, i);
        double mi0 = edge_dofs(eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid);
        m[i] = mi0;
    }

    if (deriv) {
        deriv->setZero();
    }

    if (hessian) {
        hessian->setZero();
    }

    for (int i = 0; i < m_quad_points.size(); i++) {
        double u = m_quad_points[i].u;
        double v = m_quad_points[i].v;
        double w = m_quad_points[i].weight * m_face_area[face] * m_youngs * m_thickness / 4;

        double mag = (2 * u + 2 * v - 1) * m[0] + (1 - 2 * u) * m[1] + (1 - 2 * v) * m[2];
        sum += w * (mag * mag - 1) * (mag * mag - 1);

        if (deriv || hessian) {
            Eigen::Vector3d dmag;
            dmag << 2 * u + 2 * v - 1, 1 - 2 * u, 1 - 2 * v;

            if (deriv) {
                (*deriv) += 4 * w * (mag * mag - 1) * mag * dmag;
            }

            if (hessian) {
                (*hessian) += 4 * w * (3 * mag * mag - 1) * dmag * dmag.transpose();
            }
        }
    }

    if (hessian && is_proj) {
        proj_sym_matrix(*hessian, HessianProjectType::kMaxZero);
    }
    return sum;
}

double ExtraEnergyTerms::compute_magnitude_compression_energy(const Eigen::VectorXd& edge_dofs,
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

    int nfaces = mesh.nFaces();
    for (int fid = 0; fid < nfaces; fid++) {
        Eigen::Vector3d face_deriv;
        Eigen::Matrix<double, 3, 3> face_hessian;
        double face_energy = compute_magnitude_compression_energy_perface(
            edge_dofs, mesh, fid, deriv ? &face_deriv : nullptr, hessian ? &face_hessian : nullptr, is_proj);
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

double ExtraEnergyTerms::compute_vector_perp_tangent_energy_perface(const Eigen::MatrixXd& cur_pos,
                                                                    const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::VectorXd* deriv,
                                                                    Eigen::MatrixXd* hessian,
                                                                    bool is_proj) {
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
    abar_inv = abar_inv.transpose() * abar_inv;

    for (int i = 0; i < m_quad_points.size(); i++) {
        double u = m_quad_points[i].u;
        double v = m_quad_points[i].v;
        double w = m_quad_points[i].weight * m_face_area[face] * m_shear * m_thickness / 2;

        double n_b0 = (2 * u + 2 * v - 1) * ni_b0[0] + (1 - 2 * u) * ni_b0[1] + (1 - 2 * v) * ni_b0[2];
        double n_b1 = (2 * u + 2 * v - 1) * ni_b1[0] + (1 - 2 * u) * ni_b1[1] + (1 - 2 * v) * ni_b1[2];

        Eigen::Vector2d vec;
        vec << n_b0, n_b1;

        energy += w * m_face_area[face] * vec.dot(abar_inv * vec);

        if (deriv || hessian) {
            Eigen::Vector3d bary;
            bary << 2 * u + 2 * v - 1, 1 - 2 * u, 1 - 2 * v;

            Eigen::VectorXd dn_b0 = (bary[0] * dni_b0[0] + bary[1] * dni_b0[1] + bary[2] * dni_b0[2]).transpose();
            Eigen::VectorXd dn_b1 = (bary[0] * dni_b1[0] + bary[1] * dni_b1[1] + bary[2] * dni_b1[2]).transpose();

            if (deriv) {
                (*deriv) += w * m_face_area[face] * 2.0 * abar_inv(0, 0) * n_b0 * dn_b0;
                (*deriv) += w * m_face_area[face] * abar_inv(0, 1) * (n_b0 * dn_b1 + n_b1 * dn_b0);
                (*deriv) += w * m_face_area[face] * abar_inv(1, 0) * (n_b1 * dn_b0 + n_b0 * dn_b1);
                (*deriv) += w * m_face_area[face] * 2.0 * abar_inv(1, 1) * n_b1 * dn_b1;
            }

            if (hessian) {
                Eigen::MatrixXd hn_b0 = bary[0] * hni_b0[0] + bary[1] * hni_b0[1] + bary[2] * hni_b0[2];
                Eigen::MatrixXd hn_b1 = bary[0] * hni_b1[0] + bary[1] * hni_b1[1] + bary[2] * hni_b1[2];

                (*hessian) += w * m_face_area[face] * 2.0 * abar_inv(0, 0) * (dn_b0 * dn_b0.transpose() + n_b0 * hn_b0);
                (*hessian) += w * m_face_area[face] * abar_inv(0, 1) * (dn_b0 * dn_b1.transpose() + n_b0 * hn_b1 + dn_b1 * dn_b0.transpose() + n_b1 * hn_b0);
                (*hessian) += w * m_face_area[face] * abar_inv(1, 0) * (dn_b0 * dn_b1.transpose() + n_b0 * hn_b1 + dn_b1 * dn_b0.transpose() + n_b1 * hn_b0);
                (*hessian) += w * m_face_area[face] * 2.0 * abar_inv(1, 1) * (dn_b1 * dn_b1.transpose() + n_b1 * hn_b1);
            }
        }
    }

    if (hessian && is_proj) {
        proj_sym_matrix(*hessian, HessianProjectType::kMaxZero);
    }
    return energy;
}

double ExtraEnergyTerms::compute_vector_perp_tangent_energy(const Eigen::MatrixXd& cur_pos,
                                                            const Eigen::VectorXd& edge_dofs,
                                                            const MeshConnectivity& mesh,
                                                            const std::vector<Eigen::Matrix2d> &abars,
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

    constexpr int nedgedofs = MidedgeAngleGeneralFormulation::numExtraDOFs;

    for (int i = 0; i < nfaces; i++) {
        Eigen::VectorXd face_deriv;
        Eigen::MatrixXd face_hess;

        double face_energy = compute_vector_perp_tangent_energy_perface(cur_pos, edge_dofs, mesh, abars, i, deriv ? &face_deriv : nullptr, hessian ? &face_hess : nullptr, is_proj);
        energy += face_energy;

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

double ExtraEnergyTerms::compute_magnitude_sq_change_energy_perface(const Eigen::VectorXd& edge_dofs,
                                                                    const MeshConnectivity& mesh,
                                                                    const std::vector<Eigen::Matrix2d>& abars,
                                                                    int face,
                                                                    Eigen::Vector3d* deriv,
                                                                    Eigen::Matrix<double, 3, 3>* hessian,
                                                                    bool is_proj) {
    double sum = 0;
    std::vector<double> m(3);

    for (int i = 0; i < 3; i++) {
        int eid = mesh.faceEdge(face, i);
        int efid = mesh.faceEdgeOrientation(face, i);
        double mi0 = edge_dofs(eid * MidedgeAngleGeneralFormulation::numExtraDOFs + 2 + efid);
        m[i] = mi0;
    }

    if (deriv) {
        deriv->setZero();
    }

    if (hessian) {
        hessian->setZero();
    }

    Eigen::Matrix2d abar_inv = abars[face].inverse();
    abar_inv = abar_inv.transpose() * abar_inv;

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

double ExtraEnergyTerms::compute_magnitude_sq_change_energy(const Eigen::VectorXd& edge_dofs,
                                                            const MeshConnectivity& mesh,
                                                            const std::vector<Eigen::Matrix2d>& abars,
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

    int nfaces = mesh.nFaces();
    for (int fid = 0; fid < nfaces; fid++) {
        Eigen::Vector3d face_deriv;
        Eigen::Matrix<double, 3, 3> face_hessian;
        double face_energy = compute_magnitude_sq_change_energy_perface(
            edge_dofs, mesh, abars, fid, deriv ? &face_deriv : nullptr, hessian ? &face_hessian : nullptr, is_proj);
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

void ExtraEnergyTerms::test_compute_magnitude_compression_energy_perface(const MeshConnectivity& mesh,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face) {
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
        Eigen::Vector3d dense_deriv;
        Eigen::Matrix3d dense_hess;
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

void ExtraEnergyTerms::test_compute_magnitude_compression_energy(const MeshConnectivity& mesh,
                                                                 const Eigen::VectorXd& edge_dofs) {
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

void ExtraEnergyTerms::test_compute_vector_perp_tangent_energy_perface(const MeshConnectivity& mesh,
const std::vector<Eigen::Matrix2d>& abars,
                                                                       const Eigen::MatrixXd& cur_pos,
                                                                       const Eigen::VectorXd& edge_dofs,
                                                                       int face) {
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

void ExtraEnergyTerms::test_compute_vector_perp_tangent_energy(const MeshConnectivity& mesh,
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


void ExtraEnergyTerms::test_compute_magnitude_sq_change_energy_perface(const MeshConnectivity& mesh,
                                                                       const std::vector<Eigen::Matrix2d>& abars,
                                                                         const Eigen::VectorXd& edge_dofs,
                                                                         int face) {
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
        Eigen::Vector3d dense_deriv;
        Eigen::Matrix3d dense_hess;
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

void ExtraEnergyTerms::test_compute_magnitude_sq_change_energy(const MeshConnectivity& mesh,
                                                               const std::vector<Eigen::Matrix2d>& abars,
                                                                 const Eigen::VectorXd& edge_dofs) {
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

template void proj_sym_matrix(Eigen::MatrixBase<Eigen::Matrix<double, 3, 3>>& symA, const HessianProjectType& projType);
template void proj_sym_matrix(Eigen::MatrixBase<Eigen::MatrixXd>& symA, const HessianProjectType& projType);

}  // namespace LibShell