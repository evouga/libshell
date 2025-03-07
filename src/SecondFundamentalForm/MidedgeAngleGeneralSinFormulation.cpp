//
// Created by Zhen Chen on 11/25/24.
//
#include "../../include/MidedgeAngleGeneralSinFormulation.h"
#include "../../include/MeshConnectivity.h"
#include "../GeometryDerivatives.h"

#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <iostream>
#include <random>

static void TestFuncGradHessian(
    std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> obj_Func,
    const Eigen::VectorXd& x0) {
    Eigen::VectorXd dir = x0;
    dir.setRandom();

    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> H;

    double f = obj_Func(x0, &grad, &H, false);
    std::cout << "energy: " << f << ", gradient L2-norm: " << grad.norm() << ", hessian L2-norm: " << H.norm()
              << std::endl;

    Eigen::VectorXd Hd = H * dir;
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

// Define the static member variable
constexpr int MidedgeAngleGeneralSinFormulation::numExtraDOFs;

// ni^T bj = cos(sigma) bj^T ei / |ei| + sin(sigma) sin(zeta) sij * |bj x ei| / |ei|
//         = cos(sigma) bj^T ei / |ei| + sin(sigma) sin(zeta) sij * hi, if bj is not parallel to ei
//         = cos(sigma) |ei| * sign(bj^T ei), if bj is parallel to ei
double MidedgeAngleGeneralSinFormulation::compute_nibj(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    int face,
    int i,
    int j,
    Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>* derivative,
    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>* hessian) {
    assert(i >= 0 && i < 3 && j >= 0 && j < 2);
    int efid = mesh.faceEdgeOrientation(face, i);
    assert(efid == 0 || efid == 1);
    int eid = mesh.faceEdge(face, i);

    if (derivative) {
        derivative->setZero();
    }

    if (hessian) {
        hessian->setZero();
    }
    std::vector<VectorRelationship> vector_relationships = get_edge_face_basis_relationship(mesh, eid);
    VectorRelationship& vector_relationship = vector_relationships[2 * efid + j];
    assert(vector_relationship != VectorRelationship::kUndefined);

    double sigma = edgeDOFs(eid * numExtraDOFs + 1);

    int ev[2];
    for (int k = 0; k < 3; k++) {
        if (mesh.faceVertex(face, k) == mesh.edgeVertex(eid, 0)) {
            ev[0] = k;
        }
        if (mesh.faceVertex(face, k) == mesh.edgeVertex(eid, 1)) {
            ev[1] = k;
        }
    }

    Eigen::Vector3d e = curPos.row(mesh.edgeVertex(eid, 1)) - curPos.row(mesh.edgeVertex(eid, 0));
    Eigen::Vector3d norm_deriv;
    Eigen::Matrix3d norm_hess;
    double enorm = vec_norm(e, (derivative || hessian) ? &norm_deriv : nullptr, hessian ? &norm_hess : nullptr);

    // ni^T bj = cos(sigma) |ei| * sign(bj^T ei)
    if (vector_relationship == VectorRelationship::kSameDirection ||
        vector_relationship == VectorRelationship::kOppositeDirection) {
        double sign = vector_relationship == VectorRelationship::kSameDirection ? 1.0 : -1.0;
        double res = std::cos(sigma) * enorm * sign;

        if (derivative) {
            (*derivative)(18 + i * numExtraDOFs + 1) = sign * -std::sin(sigma) * enorm;
            derivative->block<1, 3>(0, 3 * ev[0]) = -sign * std::cos(sigma) * norm_deriv;
            derivative->block<1, 3>(0, 3 * ev[1]) = sign * std::cos(sigma) * norm_deriv;
        }

        if (hessian) {
            (*hessian)(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) = -std::cos(sigma) * enorm * sign;
            hessian->block<1, 3>(18 + i * numExtraDOFs + 1, 3 * ev[0]) =
                std::sin(sigma) * sign * norm_deriv.transpose();
            hessian->block<1, 3>(18 + i * numExtraDOFs + 1, 3 * ev[1]) =
                -std::sin(sigma) * sign * norm_deriv.transpose();

            hessian->block<3, 1>(3 * ev[0], 18 + i * numExtraDOFs + 1) = std::sin(sigma) * sign * norm_deriv;
            hessian->block<3, 3>(3 * ev[0], 3 * ev[0]) = std::cos(sigma) * sign * norm_hess;
            hessian->block<3, 3>(3 * ev[0], 3 * ev[1]) = -std::cos(sigma) * sign * norm_hess;

            hessian->block<3, 1>(3 * ev[1], 18 + i * numExtraDOFs + 1) = -std::sin(sigma) * sign * norm_deriv;
            hessian->block<3, 3>(3 * ev[1], 3 * ev[1]) = std::cos(sigma) * sign * norm_hess;
            hessian->block<3, 3>(3 * ev[1], 3 * ev[0]) = -std::cos(sigma) * sign * norm_hess;
        }

        return res;
    } else {
        // ni^T bj = cos(sigma) bj^T ei / |ei| + sin(sigma) sin(zeta) sij * hi, if bj is not parallel to ei
        Eigen::Vector3d bj = curPos.row(mesh.faceVertex(face, (j + 1) % 3)) - curPos.row(mesh.faceVertex(face, 0));
        double dot_prod = bj.dot(e);
        double dot_over_norm = dot_prod / enorm;
        double part1 = std::cos(sigma) * dot_over_norm;

        Eigen::Matrix<double, 1, 12> thetaderiv;
        Eigen::Matrix<double, 12, 12> thetahess;
        double theta = edgeTheta(mesh, curPos, eid, (derivative || hessian) ? &thetaderiv : nullptr,
                                 hessian ? &thetahess : nullptr);

        Eigen::Matrix<double, 1, 9> hderiv;
        Eigen::Matrix<double, 9, 9> hhess;
        double altitude = triangleAltitude(mesh, curPos, face, i, (derivative || hessian) ? &hderiv : nullptr,
                                           hessian ? &hhess : nullptr);

        double orient = mesh.faceEdgeOrientation(face, i) == 0 ? 1.0 : -1.0;
        double zeta = orient * 0.5 * theta + edgeDOFs[numExtraDOFs * eid];
        double sij = vector_relationship == VectorRelationship::kPositiveOrientation ? 1.0 : -1.0;
        double part2 = sij * std::sin(sigma) * std::sin(zeta) * altitude;

        if (derivative || hessian) {
            // derivatives and hessian from first part
            {
                // cos(sigma) bj^T ei / |ei|
                Eigen::Matrix<double, 1, 9> norm_deriv_full;
                norm_deriv_full.setZero();
                norm_deriv_full.block<1, 3>(0, 3 * ev[0]) = -norm_deriv;
                norm_deriv_full.block<1, 3>(0, 3 * ev[1]) = norm_deriv;

                Eigen::Matrix<double, 1, 9> dot_deriv;
                Eigen::Matrix<double, 3, 9> grad_e = Eigen::Matrix<double, 3, 9>::Zero();
                Eigen::Matrix<double, 3, 9> grad_b = Eigen::Matrix<double, 3, 9>::Zero();

                grad_e.block<3, 3>(0, 3 * ev[0]).setIdentity();
                grad_e.block<3, 3>(0, 3 * ev[0]) *= -1;
                grad_e.block<3, 3>(0, 3 * ev[1]).setIdentity();

                grad_b.block<3, 3>(0, 0).setIdentity();
                grad_b.block<3, 3>(0, 0) *= -1;
                grad_b.block<3, 3>(0, 3 * ((j + 1) % 3)).setIdentity();
                dot_deriv = bj.transpose() * grad_e + e.transpose() * grad_b;

                Eigen::Matrix<double, 1, 9> dot_over_norm_deriv =
                    dot_deriv / enorm - dot_prod * norm_deriv_full / (enorm * enorm);

                if (derivative) {
                    (*derivative)(18 + i * numExtraDOFs + 1) += -dot_over_norm * std::sin(sigma);
                    derivative->block<1, 9>(0, 0) += std::cos(sigma) * dot_over_norm_deriv;
                }

                if (hessian) {
                    Eigen::Matrix<double, 9, 9> norm_hess_full;
                    norm_hess_full.setZero();
                    norm_hess_full.block<3, 3>(3 * ev[0], 3 * ev[0]) = norm_hess;
                    norm_hess_full.block<3, 3>(3 * ev[0], 3 * ev[1]) = -norm_hess;
                    norm_hess_full.block<3, 3>(3 * ev[1], 3 * ev[0]) = -norm_hess;
                    norm_hess_full.block<3, 3>(3 * ev[1], 3 * ev[1]) = norm_hess;

                    Eigen::Matrix<double, 9, 9> dot_hess;
                    dot_hess = grad_b.transpose() * grad_e + grad_e.transpose() * grad_b;

                    Eigen::Matrix<double, 9, 9> dot_over_norm_hess =
                        dot_hess / enorm -
                        (dot_deriv.transpose() * norm_deriv_full + norm_deriv_full.transpose() * dot_deriv) /
                            (enorm * enorm) -
                        dot_prod * norm_hess_full / (enorm * enorm) +
                        2 * dot_prod * norm_deriv_full.transpose() * norm_deriv_full / (enorm * enorm * enorm);

                    (*hessian)(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                        -dot_over_norm * std::cos(sigma);
                    hessian->block<1, 9>(18 + i * numExtraDOFs + 1, 0) += -std::sin(sigma) * dot_over_norm_deriv;

                    hessian->block<9, 9>(0, 0) += std::cos(sigma) * dot_over_norm_hess;
                    hessian->block<9, 1>(0, 18 + i * numExtraDOFs + 1) +=
                        -std::sin(sigma) * dot_over_norm_deriv.transpose();
                }
            }

            {
                int av[4];
                if (mesh.faceEdgeOrientation(face, i) == 0) {
                    av[0] = (i + 1) % 3;
                    av[1] = (i + 2) % 3;
                    av[2] = i;
                    av[3] = 3 + i;
                } else {
                    av[0] = (i + 2) % 3;
                    av[1] = (i + 1) % 3;
                    av[2] = 3 + i;
                    av[3] = i;
                }

                int hv[3];
                for (int k = 0; k < 3; k++) {
                    hv[k] = (i + k) % 3;
                }

                // sij * std::sin(sigma) * std::sin(zeta) * altitude;
                double sin_zeta_altitude = std::sin(zeta) * altitude;
                Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> sin_zeta_altitude_deriv;
                sin_zeta_altitude_deriv.setZero();

                for (int k = 0; k < 3; k++) {
                    sin_zeta_altitude_deriv.block<1, 3>(0, 3 * hv[k]) += std::sin(zeta) * hderiv.block<1, 3>(0, 3 * k);
                }

                for (int k = 0; k < 4; k++) {
                    sin_zeta_altitude_deriv.block<1, 3>(0, 3 * av[k]) +=
                        orient * 0.5 * altitude * std::cos(zeta) * thetaderiv.block<1, 3>(0, 3 * k);
                }
                sin_zeta_altitude_deriv(0, 18 + i * numExtraDOFs) += altitude * std::cos(zeta);

                if (derivative) {
                    (*derivative) += sij * std::sin(sigma) * sin_zeta_altitude_deriv;
                    (*derivative)(0, 18 + i * numExtraDOFs + 1) += sij * std::cos(sigma) * sin_zeta_altitude;
                }

                if (hessian) {
                    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> sin_altitude_hess;
                    sin_altitude_hess.setZero();

                    for (int m = 0; m < 3; m++) {
                        for (int k = 0; k < 3; k++) {
                            sin_altitude_hess.block<3, 3>(3 * hv[m], 3 * hv[k]) +=
                                std::sin(zeta) * hhess.block<3, 3>(3 * m, 3 * k);
                        }
                    }

                    for (int k = 0; k < 3; k++) {
                        for (int m = 0; m < 4; m++) {
                            sin_altitude_hess.block<3, 3>(3 * av[m], 3 * hv[k]) +=
                                orient * 0.5 * std::cos(zeta) * thetaderiv.block(0, 3 * m, 1, 3).transpose() *
                                hderiv.block(0, 3 * k, 1, 3);
                            sin_altitude_hess.block<3, 3>(3 * hv[k], 3 * av[m]) +=
                                orient * 0.5 * std::cos(zeta) * hderiv.block(0, 3 * k, 1, 3).transpose() *
                                thetaderiv.block(0, 3 * m, 1, 3);
                        }
                        sin_altitude_hess.block<1, 3>(18 + i * numExtraDOFs, 3 * hv[k]) +=
                            std::cos(zeta) * hderiv.block<1, 3>(0, 3 * k);
                        sin_altitude_hess.block<3, 1>(3 * hv[k], 18 + i * numExtraDOFs) +=
                            std::cos(zeta) * hderiv.block<1, 3>(0, 3 * k).transpose();
                    }

                    for (int k = 0; k < 4; k++) {
                        for (int m = 0; m < 4; m++) {
                            sin_altitude_hess.block<3, 3>(3 * av[m], 3 * av[k]) +=
                                orient * 0.5 * altitude * std::cos(zeta) * thetahess.block<3, 3>(3 * m, 3 * k);
                            sin_altitude_hess.block<3, 3>(3 * av[m], 3 * av[k]) +=
                                -0.25 * altitude * std::sin(zeta) * thetaderiv.block<1, 3>(0, 3 * m).transpose() *
                                thetaderiv.block<1, 3>(0, 3 * k);
                        }
                        sin_altitude_hess.block<1, 3>(18 + i * numExtraDOFs, 3 * av[k]) +=
                            -0.5 * altitude * std::sin(zeta) * orient * thetaderiv.block<1, 3>(0, 3 * k);
                        sin_altitude_hess.block<3, 1>(3 * av[k], 18 + i * numExtraDOFs) +=
                            -0.5 * altitude * std::sin(zeta) * orient * thetaderiv.block<1, 3>(0, 3 * k).transpose();
                    }

                    sin_altitude_hess(18 + i * numExtraDOFs, 18 + i * numExtraDOFs) += -altitude * std::sin(zeta);

                    (*hessian)(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                        -sij * std::sin(sigma) * sin_zeta_altitude;
                    hessian->block<1, 18 + 3 * numExtraDOFs>(18 + i * numExtraDOFs + 1, 0) +=
                        sij * std::cos(sigma) * sin_zeta_altitude_deriv;

                    (*hessian) += sij * std::sin(sigma) * sin_altitude_hess;
                    hessian->block<18 + 3 * numExtraDOFs, 1>(0, 18 + i * numExtraDOFs + 1) +=
                        sij * std::cos(sigma) * sin_zeta_altitude_deriv.transpose();
                }
            }
        }

        return part1 + part2;
    }
}

double MidedgeAngleGeneralSinFormulation::compute_niei(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    int face,
    int i,
    Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>* derivative,
    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>* hessian) {
    assert(i >= 0 && i < 3);
    int efid = mesh.faceEdgeOrientation(face, i);
    assert(efid == 0 || efid == 1);
    int eid = mesh.faceEdge(face, i);

    if (derivative) {
        derivative->setZero();
    }

    if (hessian) {
        hessian->setZero();
    }

    double sigma = edgeDOFs(eid * numExtraDOFs + 1);

    int ev[2];
    for (int k = 0; k < 3; k++) {
        if (mesh.faceVertex(face, k) == mesh.edgeVertex(eid, 0)) {
            ev[0] = k;
        }
        if (mesh.faceVertex(face, k) == mesh.edgeVertex(eid, 1)) {
            ev[1] = k;
        }
    }

    Eigen::Vector3d e = curPos.row(mesh.edgeVertex(eid, 1)) - curPos.row(mesh.edgeVertex(eid, 0));
    Eigen::Vector3d norm_deriv;
    Eigen::Matrix3d norm_hess;
    double enorm = vec_norm(e, (derivative || hessian) ? &norm_deriv : nullptr, hessian ? &norm_hess : nullptr);

    // ni^T bj = cos(sigma) |ei|
    double sign = 1.0;
    double res = std::cos(sigma) * enorm * sign;

    if (derivative) {
        (*derivative)(18 + i * numExtraDOFs + 1) = sign * -std::sin(sigma) * enorm;
        derivative->block<1, 3>(0, 3 * ev[0]) = -sign * std::cos(sigma) * norm_deriv;
        derivative->block<1, 3>(0, 3 * ev[1]) = sign * std::cos(sigma) * norm_deriv;
    }

    if (hessian) {
        (*hessian)(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) = -std::cos(sigma) * enorm * sign;
        hessian->block<1, 3>(18 + i * numExtraDOFs + 1, 3 * ev[0]) = std::sin(sigma) * sign * norm_deriv.transpose();
        hessian->block<1, 3>(18 + i * numExtraDOFs + 1, 3 * ev[1]) = -std::sin(sigma) * sign * norm_deriv.transpose();

        hessian->block<3, 1>(3 * ev[0], 18 + i * numExtraDOFs + 1) = std::sin(sigma) * sign * norm_deriv;
        hessian->block<3, 3>(3 * ev[0], 3 * ev[0]) = std::cos(sigma) * sign * norm_hess;
        hessian->block<3, 3>(3 * ev[0], 3 * ev[1]) = -std::cos(sigma) * sign * norm_hess;

        hessian->block<3, 1>(3 * ev[1], 18 + i * numExtraDOFs + 1) = -std::sin(sigma) * sign * norm_deriv;
        hessian->block<3, 3>(3 * ev[1], 3 * ev[1]) = std::cos(sigma) * sign * norm_hess;
        hessian->block<3, 3>(3 * ev[1], 3 * ev[0]) = -std::cos(sigma) * sign * norm_hess;
    }

    return res;
}

double MidedgeAngleGeneralSinFormulation::compute_ninj(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    int face,
    int i,
    int j,
    Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>* derivative,
    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>* hessian) {
    // ni = mi[cos(σi) êi + sin(σi) cos(ζi) nf + sin(σi) sin(ζi) (êi x nf)]
    // nj = mj[cos(σj) êj + sin(σj) cos(ζj) nf + sin(σj) sin(ζj) (êj x nf)]
    // for sin formulation mi = mj = 1

    // initialization
    double ninj = 0;

    if (derivative) {
        derivative->setZero();
    }

    if (hessian) {
        hessian->setZero();
    }

    // special case i = j
    if (i == j) {
        // ninj = cos(σi)^2 + sin(σi)^2 cos(ζi)^2 + sin(σi)^2 sin(ζi)^2 = 1
        ninj = 1.0;
    } else {
        Eigen::Matrix<double, 3, 9> nf_deriv;
        std::vector<Eigen::Matrix<double, 9, 9>> nf_hess;
        Eigen::Vector3d nf = faceNormal(mesh, curPos, face, 0, (derivative || hessian) ? &nf_deriv : nullptr,
                                        hessian ? &nf_hess : nullptr);
        if (nf.norm() == 0) {
            return 0;
        }

        int edgeIdx_i = mesh.faceEdge(face, i);
        int edgeIdx_j = mesh.faceEdge(face, j);
        Eigen::Vector3d ei = curPos.row(mesh.edgeVertex(edgeIdx_i, 1)) - curPos.row(mesh.edgeVertex(edgeIdx_i, 0));
        Eigen::Vector3d ej = curPos.row(mesh.edgeVertex(edgeIdx_j, 1)) - curPos.row(mesh.edgeVertex(edgeIdx_j, 0));

        int eiv[2] = {-1, -1}, ejv[2] = {-1, -1};
        for (int k = 0; k < 3; k++) {
            if (mesh.faceVertex(face, k) == mesh.edgeVertex(edgeIdx_i, 0)) {
                eiv[0] = k;
            }
            if (mesh.faceVertex(face, k) == mesh.edgeVertex(edgeIdx_i, 1)) {
                eiv[1] = k;
            }

            if (mesh.faceVertex(face, k) == mesh.edgeVertex(edgeIdx_j, 0)) {
                ejv[0] = k;
            }
            if (mesh.faceVertex(face, k) == mesh.edgeVertex(edgeIdx_j, 1)) {
                ejv[1] = k;
            }
        }

        assert(eiv[0] != -1 && eiv[1] != -1);
        assert(ejv[0] != -1 && ejv[1] != -1);

        int common_vertex = -1;
        for (int k = 0; k < 3; k++) {
            if (k != i && k != j) {
                common_vertex = k;
                break;
            }
        }

        // cosine of the rotation angle from ei to ej with face_normal as the rotation axis
        Eigen::Matrix<double, 1, 9> cos_wij_deriv, sin_wij_deriv;
        Eigen::Matrix<double, 9, 9> cos_wij_hess, sin_wij_hess;
        double cos_wij =
            cosTriangleAngle(mesh, curPos, face, common_vertex, (derivative || hessian) ? &cos_wij_deriv : nullptr,
                             hessian ? &cos_wij_hess : nullptr);
        double sin_wij = std::sqrt(1 - cos_wij * cos_wij);
        if (derivative || hessian) {
            double same_divider = std::max(sin_wij, 1e-8);
            sin_wij_deriv = -cos_wij / same_divider * cos_wij_deriv;

            if (hessian) {
                sin_wij_hess = -cos_wij / same_divider * cos_wij_hess -
                               1 / (same_divider * same_divider * same_divider) * cos_wij_deriv.transpose() * cos_wij_deriv;
            }
        }

        // need to determine the sign of the rotation angle
        int cos_sign_ij = 1, sin_sign_ij = 1;
        {
            // the triangle angle is the angle rotate {V_{common+1}, V_{common}} to {V_{common + 1}, V_{common}} w.r.t.
            // triangle normal whether ei (ej) is V_{common + 1} - V_{common} or V_{common + 2} - V_{common}
            int sign_i = mesh.edgeVertex(edgeIdx_i, 0) == mesh.faceVertex(face, common_vertex) ? 1 : -1;
            int sign_j = mesh.edgeVertex(edgeIdx_j, 0) == mesh.faceVertex(face, common_vertex) ? 1 : -1;

            // edge direction matches the face edge direction
            if (sign_i == 1 && sign_j == 1) {
                if (i == (common_vertex + 2) % 3) {
                    assert(j == (common_vertex + 1) % 3);
                    // In this case, the ei = {V_{common + 1}, V_{common}}, ej = {V_{common + 2}, V_{common}};
                    // rotation angle = triangle angle
                    cos_sign_ij = 1;
                    sin_sign_ij = 1;
                } else {
                    assert(j == (common_vertex + 2) % 3 && i == (common_vertex + 1) % 3);
                    // In this case, the ei = {V_{common + 2}, V_{common}}, ej = {V_{common + 1}, V_{common}};
                    // rotation angle = -triangle angle
                    cos_sign_ij = 1;
                    sin_sign_ij = -1;
                }
            }
            // edge i matches the face edge direction, but edge j not
            if (sign_i == 1 && sign_j == -1) {
                if (i == (common_vertex + 2) % 3) {
                    assert(j == (common_vertex + 1) % 3);
                    // In this case, the ei = {V_{common + 1}, V_{common}}, ej = {V_{common}, V_{common + 2}};
                    // rotation angle = triangle angle + 180 degree
                    cos_sign_ij = -1;
                    sin_sign_ij = -1;
                } else {
                    assert(j == (common_vertex + 2) % 3 && i == (common_vertex + 1) % 3);
                    // In this case, the ei = {V_{common + 2}, V_{common}}, ej = {V_{common}, V_{common + 1}};
                    // rotation angle = 180 degree - triangle angle
                    cos_sign_ij = -1;
                    sin_sign_ij = 1;
                }
            }
            // edge j matches the face edge direction, but edge i not
            if (sign_i == -1 && sign_j == 1) {
                if (i == (common_vertex + 2) % 3) {
                    assert(j == (common_vertex + 1) % 3);
                    // In this case, the ei = { V_{common}, V_{common + 1}}, ej = {V_{common + 2}, V_{common}};
                    // rotation angle = triangle angle + 180 degree
                    cos_sign_ij = -1;
                    sin_sign_ij = -1;
                } else {
                    assert(j == (common_vertex + 2) % 3 && i == (common_vertex + 1) % 3);
                    // In this case, the ei = {V_{common}, V_{common + 2}}, ej = {V_{common + 1}, V_{common}};
                    // rotation angle = 180 degree - triangle angle
                    cos_sign_ij = -1;
                    sin_sign_ij = 1;
                }
            }
            // edge direction does not match the face edge direction
            if (sign_i == -1 && sign_j == -1) {
                if (i == (common_vertex + 2) % 3) {
                    assert(j == (common_vertex + 1) % 3);
                    // In this case, the ei = { V_{common}, V_{common + 1}}, ej = {V_{common}, V_{common + 2}};
                    // rotation angle = triangle angle
                    cos_sign_ij = 1;
                    sin_sign_ij = 1;
                } else {
                    assert(j == (common_vertex + 2) % 3 && i == (common_vertex + 1) % 3);
                    // In this case, the ei = {V_{common}, V_{common + 2}}, ej = {V_{common}, V_{common + 1}};
                    // rotation angle = -triangle angle
                    cos_sign_ij = 1;
                    sin_sign_ij = -1;
                }
            }
        }

        cos_wij *= cos_sign_ij;
        sin_wij *= sin_sign_ij;

        if (derivative || hessian) {
            cos_wij_deriv *= cos_sign_ij;
            sin_wij_deriv *= sin_sign_ij;
            if (hessian) {
                cos_wij_hess *= cos_sign_ij;
                sin_wij_hess *= sin_sign_ij;
            }
        }

        // ni = cos(σi) êi + sin(σi) cos(ζi) nf + sin(σi) sin(ζi) (êi x nf) = ai êi + bi nf + ci (êi x nf)
        // nj = cos(σj) êj + sin(σj) cos(ζj) nf + sin(σj) sin(ζj) (êj x nf) = aj êj + bj nf + cj (êj x nf)
        // ni^T nj = (ai aj + ci cj) * cos(wij) + (ai cj - aj ci) * sin(wij) + bi bj
        double sigma_i = edgeDOFs(edgeIdx_i * numExtraDOFs + 1);
        double sigma_j = edgeDOFs(edgeIdx_j * numExtraDOFs + 1);

        auto get_zeta = [&mesh, &curPos, &edgeDOFs, &face](
                            int face_eid, Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>* deriv,
                            Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>* hess) {
            int edge_id = mesh.faceEdge(face, face_eid);
            Eigen::Matrix<double, 1, 12> dtheta_dx;
            Eigen::Matrix<double, 12, 12> d2theta_dx2;
            double theta =
                edgeTheta(mesh, curPos, edge_id, deriv ? &dtheta_dx : nullptr, hess ? &d2theta_dx2 : nullptr);
            double orient = mesh.faceEdgeOrientation(face, face_eid) == 0 ? 1.0 : -1.0;
            double zeta = orient * 0.5 * theta + edgeDOFs[numExtraDOFs * edge_id];

            int av[4];  // angle vertex map
            if (mesh.faceEdgeOrientation(face, face_eid) == 0) {
                av[0] = (face_eid + 1) % 3;
                av[1] = (face_eid + 2) % 3;
                av[2] = face_eid;
                av[3] = 3 + face_eid;
            } else {
                av[0] = (face_eid + 2) % 3;
                av[1] = (face_eid + 1) % 3;
                av[2] = 3 + face_eid;
                av[3] = face_eid;
            }

            if (deriv) {
                deriv->setZero();
                (*deriv)(0, 18 + face_eid * numExtraDOFs) = 1;

                for (int k = 0; k < 4; k++) {
                    deriv->block<1, 3>(0, 3 * av[k]) = orient * 0.5 * dtheta_dx.block<1, 3>(0, 3 * k);
                }
            }

            if (hess) {
                hess->setZero();
                for (int k = 0; k < 4; k++) {
                    for (int l = 0; l < 4; l++) {
                        hess->block<3, 3>(3 * av[k], 3 * av[l]) = orient * 0.5 * d2theta_dx2.block<3, 3>(3 * k, 3 * l);
                    }
                }
            }

            return zeta;
        };

        Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> zeta_i_deriv;
        Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> zeta_i_hess;
        double zeta_i =
            get_zeta(i, (derivative || hessian) ? &zeta_i_deriv : nullptr, hessian ? &zeta_i_hess : nullptr);

        Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> zeta_j_deriv;
        Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> zeta_j_hess;
        double zeta_j =
            get_zeta(j, (derivative || hessian) ? &zeta_j_deriv : nullptr, hessian ? &zeta_j_hess : nullptr);

        double ai = std::cos(sigma_i);
        double bi = std::sin(sigma_i) * std::cos(zeta_i);
        double ci = std::sin(sigma_i) * std::sin(zeta_i);

        double aj = std::cos(sigma_j);
        double bj = std::sin(sigma_j) * std::cos(zeta_j);
        double cj = std::sin(sigma_j) * std::sin(zeta_j);

        // f0 = cos(σi) * cos(σj) + sin(σi) * sin(σj) * sin(ζi) * sin(ζj)
        double f0 = ai * aj + ci * cj;
        // f1 = cos(σi) * sin(σj) * sin(ζj) - sin(σi) * cos(σj) * sin(ζi)
        double f1 = ai * cj - aj * ci;
        // f2 = sin(σi) * sin(σj) * cos(ζi) * cos(ζj)
        double f2 = bi * bj;

        ninj = f0 * cos_wij + f1 * sin_wij + f2;

        if (derivative || hessian) {
            Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> f0_deriv, f1_deriv, f2_deriv;

            // dzeta/dsigma = 0
            // f0 = cos(σi) * cos(σj) + sin(σi) * sin(σj) * sin(ζi) * sin(ζj)
            f0_deriv = std::sin(sigma_i) * std::sin(sigma_j) *
                       (std::sin(zeta_i) * std::cos(zeta_j) * zeta_j_deriv +
                        std::sin(zeta_j) * std::cos(zeta_i) * zeta_i_deriv);
            f0_deriv[18 + i * numExtraDOFs + 1] +=
                -std::sin(sigma_i) * std::cos(sigma_j) + std::cos(sigma_i) * std::sin(zeta_i) * cj;
            f0_deriv[18 + j * numExtraDOFs + 1] +=
                -std::sin(sigma_j) * std::cos(sigma_i) + std::cos(sigma_j) * std::sin(zeta_j) * ci;

            // f1 = cos(σi) * sin(σj) * sin(ζj) - sin(σi) * cos(σj) * sin(ζi)
            f1_deriv = std::cos(sigma_i) * std::sin(sigma_j) * std::cos(zeta_j) * zeta_j_deriv -
                       std::sin(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * zeta_i_deriv;
            f1_deriv[18 + i * numExtraDOFs + 1] += -std::sin(sigma_i) * std::sin(sigma_j) * std::sin(zeta_j) -
                                                   std::cos(sigma_i) * std::cos(sigma_j) * std::sin(zeta_i);
            f1_deriv[18 + j * numExtraDOFs + 1] += std::cos(sigma_i) * std::cos(sigma_j) * std::sin(zeta_j) +
                                                   std::sin(sigma_i) * std::sin(sigma_j) * std::sin(zeta_i);

            // f2 = sin(σi) * sin(σj) * cos(ζi) * cos(ζj)
            f2_deriv = std::sin(sigma_i) * std::sin(sigma_j) *
                       (-std::sin(zeta_i) * std::cos(zeta_j) * zeta_i_deriv -
                        std::cos(zeta_i) * std::sin(zeta_j) * zeta_j_deriv);
            f2_deriv[18 + i * numExtraDOFs + 1] +=
                std::cos(sigma_i) * std::sin(sigma_j) * std::cos(zeta_i) * std::cos(zeta_j);
            f2_deriv[18 + j * numExtraDOFs + 1] +=
                std::sin(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * std::cos(zeta_j);

            if (derivative) {
                // ninj = f0 * std::cos(wij) + f1 * std::sin(wij) + f2;
                *derivative = cos_wij * f0_deriv + sin_wij * f1_deriv + f2_deriv;
                derivative->block<1, 9>(0, 0) += f0 * cos_wij_deriv + f1 * sin_wij_deriv;
            }

            if (hessian) {
                Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> f0_hess, f1_hess, f2_hess;

                f0_hess = std::sin(sigma_i) * std::sin(sigma_j) *
                          (std::sin(zeta_i) * std::cos(zeta_j) * zeta_j_hess +
                           std::sin(zeta_j) * std::cos(zeta_i) * zeta_i_hess);
                f0_hess += -std::sin(sigma_i) * std::sin(sigma_j) * std::sin(zeta_i) * std::sin(zeta_j) *
                           (zeta_i_deriv.transpose() * zeta_i_deriv + zeta_j_deriv.transpose() * zeta_j_deriv);
                f0_hess += std::sin(sigma_i) * std::sin(sigma_j) * std::cos(zeta_i) * std::cos(zeta_j) *
                           (zeta_i_deriv.transpose() * zeta_j_deriv + zeta_j_deriv.transpose() * zeta_i_deriv);

                f0_hess(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) += -f0;
                f0_hess.row(18 + i * numExtraDOFs + 1) += std::cos(sigma_i) * std::sin(sigma_j) *
                                                          (std::sin(zeta_i) * std::cos(zeta_j) * zeta_j_deriv +
                                                           std::sin(zeta_j) * std::cos(zeta_i) * zeta_i_deriv);
                f0_hess.col(18 + i * numExtraDOFs + 1) += std::cos(sigma_i) * std::sin(sigma_j) *
                                                          (std::sin(zeta_i) * std::cos(zeta_j) * zeta_j_deriv +
                                                           std::sin(zeta_j) * std::cos(zeta_i) * zeta_i_deriv)
                                                              .transpose();
                f0_hess(18 + i * numExtraDOFs + 1, 18 + j * numExtraDOFs + 1) +=
                    std::sin(sigma_i) * std::sin(sigma_j) +
                    std::cos(sigma_i) * std::cos(sigma_j) * std::sin(zeta_i) * std::sin(zeta_j);

                f0_hess(18 + j * numExtraDOFs + 1, 18 + j * numExtraDOFs + 1) += -f0;
                f0_hess.row(18 + j * numExtraDOFs + 1) += std::cos(sigma_j) * std::sin(sigma_i) *
                                                          (std::sin(zeta_i) * std::cos(zeta_j) * zeta_j_deriv +
                                                           std::sin(zeta_j) * std::cos(zeta_i) * zeta_i_deriv);
                f0_hess.col(18 + j * numExtraDOFs + 1) += std::cos(sigma_j) * std::sin(sigma_i) *
                                                          (std::sin(zeta_i) * std::cos(zeta_j) * zeta_j_deriv +
                                                           std::sin(zeta_j) * std::cos(zeta_i) * zeta_i_deriv)
                                                              .transpose();
                f0_hess(18 + j * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                    std::sin(sigma_i) * std::sin(sigma_j) +
                    std::cos(sigma_j) * std::cos(sigma_i) * std::sin(zeta_i) * std::sin(zeta_j);

                // f1 = cos(σi) * sin(σj) * sin(ζj) - sin(σi) * cos(σj) * sin(ζi)
                f1_hess = std::cos(sigma_i) * std::sin(sigma_j) * std::cos(zeta_j) * zeta_j_hess -
                          std::sin(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * zeta_i_hess;
                f1_hess +=
                    -std::cos(sigma_i) * std::sin(sigma_j) * std::sin(zeta_j) * zeta_j_deriv.transpose() *
                        zeta_j_deriv +
                    std::sin(sigma_i) * std::cos(sigma_j) * std::sin(zeta_i) * zeta_i_deriv.transpose() * zeta_i_deriv;

                f1_hess(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                    -std::cos(sigma_i) * std::sin(sigma_j) * std::sin(zeta_j) +
                    std::sin(sigma_i) * std::cos(sigma_j) * std::sin(zeta_i);
                f1_hess.row(18 + i * numExtraDOFs + 1) +=
                    -std::sin(sigma_i) * std::sin(sigma_j) * std::cos(zeta_j) * zeta_j_deriv -
                    std::cos(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * zeta_i_deriv;
                f1_hess.col(18 + i * numExtraDOFs + 1) +=
                    -std::sin(sigma_i) * std::sin(sigma_j) * std::cos(zeta_j) * zeta_j_deriv.transpose() -
                    std::cos(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * zeta_i_deriv.transpose();

                f1_hess(18 + j * numExtraDOFs + 1, 18 + j * numExtraDOFs + 1) +=
                    -std::cos(sigma_i) * std::sin(sigma_j) * std::sin(zeta_j) +
                    std::sin(sigma_i) * std::cos(sigma_j) * std::sin(zeta_i);
                f1_hess.row(18 + j * numExtraDOFs + 1) +=
                    std::cos(sigma_i) * std::cos(sigma_j) * std::cos(zeta_j) * zeta_j_deriv +
                    std::sin(sigma_i) * std::sin(sigma_j) * std::cos(zeta_i) * zeta_i_deriv;
                f1_hess.col(18 + j * numExtraDOFs + 1) +=
                    std::cos(sigma_i) * std::cos(sigma_j) * std::cos(zeta_j) * zeta_j_deriv.transpose() +
                    std::sin(sigma_i) * std::sin(sigma_j) * std::cos(zeta_i) * zeta_i_deriv.transpose();

                f1_hess(18 + i * numExtraDOFs + 1, 18 + j * numExtraDOFs + 1) +=
                    -std::sin(sigma_i) * std::cos(sigma_j) * std::sin(zeta_j) +
                    std::cos(sigma_i) * std::sin(sigma_j) * std::sin(zeta_i);
                f1_hess(18 + j * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                    -std::sin(sigma_i) * std::cos(sigma_j) * std::sin(zeta_j) +
                    std::cos(sigma_i) * std::sin(sigma_j) * std::sin(zeta_i);

                // f2 = sin(σi) * sin(σj) * cos(ζi) * cos(ζj)
                f2_hess = -std::sin(sigma_i) * std::sin(sigma_j) *
                          (std::sin(zeta_i) * std::cos(zeta_j) * zeta_i_hess +
                           std::cos(zeta_i) * std::sin(zeta_j) * zeta_j_hess);
                f2_hess += -std::sin(sigma_i) * std::sin(sigma_j) * std::cos(zeta_i) * std::cos(zeta_j) *
                           (zeta_i_deriv.transpose() * zeta_i_deriv + zeta_j_deriv.transpose() * zeta_j_deriv);
                f2_hess += std::sin(sigma_i) * std::sin(sigma_j) * std::sin(zeta_i) * std::sin(zeta_j) *
                           (zeta_i_deriv.transpose() * zeta_j_deriv + zeta_j_deriv.transpose() * zeta_i_deriv);

                f2_hess(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) += -f2;
                f2_hess.row(18 + i * numExtraDOFs + 1) += std::cos(sigma_i) * std::sin(sigma_j) *
                                                          (-std::sin(zeta_i) * std::cos(zeta_j) * zeta_i_deriv -
                                                           std::cos(zeta_i) * std::sin(zeta_j) * zeta_j_deriv);
                f2_hess.col(18 + i * numExtraDOFs + 1) += std::cos(sigma_i) * std::sin(sigma_j) *
                                                          (-std::sin(zeta_i) * std::cos(zeta_j) * zeta_i_deriv -
                                                           std::cos(zeta_i) * std::sin(zeta_j) * zeta_j_deriv)
                                                              .transpose();
                f2_hess(18 + i * numExtraDOFs + 1, 18 + j * numExtraDOFs + 1) +=
                    std::cos(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * std::cos(zeta_j);

                f2_hess(18 + j * numExtraDOFs + 1, 18 + j * numExtraDOFs + 1) += -f2;
                f2_hess.row(18 + j * numExtraDOFs + 1) += std::sin(sigma_i) * std::cos(sigma_j) *
                                                          (-std::sin(zeta_i) * std::cos(zeta_j) * zeta_i_deriv -
                                                           std::cos(zeta_i) * std::sin(zeta_j) * zeta_j_deriv);
                f2_hess.col(18 + j * numExtraDOFs + 1) += std::sin(sigma_i) * std::cos(sigma_j) *
                                                          (-std::sin(zeta_i) * std::cos(zeta_j) * zeta_i_deriv -
                                                           std::cos(zeta_i) * std::sin(zeta_j) * zeta_j_deriv)
                                                              .transpose();
                f2_hess(18 + j * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                    std::cos(sigma_i) * std::cos(sigma_j) * std::cos(zeta_i) * std::cos(zeta_j);

                // ninj = f0 * std::cos(wij) + f1 * std::sin(wij) + f2;
                *hessian = cos_wij * f0_hess + sin_wij * f1_hess + f2_hess;
                hessian->block<9, 18 + 3 * numExtraDOFs>(0, 0) +=
                    cos_wij_deriv.transpose() * f0_deriv + sin_wij_deriv.transpose() * f1_deriv;
                hessian->block<18 + 3 * numExtraDOFs, 9>(0, 0) +=
                    f0_deriv.transpose() * cos_wij_deriv + f1_deriv.transpose() * sin_wij_deriv;
                hessian->block<9, 9>(0, 0) += f0 * cos_wij_hess + f1 * sin_wij_hess;
            }
        }
    }

    return ninj;
}

Eigen::Matrix2d MidedgeAngleGeneralSinFormulation::secondFundamentalForm(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& extraDOFs,
    int face,
    Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs>* derivative,
    std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>>* hessian) {
    if (derivative) {
        derivative->setZero();
    }
    if (hessian) {
        hessian->resize(4);
        for (int i = 0; i < 4; i++) {
            (*hessian)[i].setZero();
        }
    }

    std::vector<double> nibj(6);
    std::vector<Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>> nibj_deriv(6);
    std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>> nibj_hessian(6);

    for (int i = 0; i < 6; i++) {
        int edge = i / 2;
        int bid = i % 2;
        nibj[i] = compute_nibj(mesh, curPos, extraDOFs, face, edge, bid, derivative ? &nibj_deriv[i] : nullptr,
                               hessian ? &nibj_hessian[i] : nullptr);
    }

    auto get_idx = [&](int i, int j) -> int { return i * 2 + j; };

    Eigen::Matrix2d II;
    // first entry: II(0, 0) = -2(n1^T b0 - n0^T b0)
    II(0, 0) = -2 * (nibj[get_idx(1, 0)] - nibj[get_idx(0, 0)]);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(0, 0) =
            -2 * (nibj_deriv[get_idx(1, 0)] - nibj_deriv[get_idx(0, 0)]);
    }
    if (hessian) {
        (*hessian)[0] = -2 * (nibj_hessian[get_idx(1, 0)] - nibj_hessian[get_idx(0, 0)]);
    }

    // second entry: II(0, 1) = -(n1^T b1 - n0^T b1) - (n2^T b0 - n0^T b0)
    II(0, 1) = -(nibj[get_idx(1, 1)] - nibj[get_idx(0, 1)]) - (nibj[get_idx(2, 0)] - nibj[get_idx(0, 0)]);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0) = -nibj_deriv[get_idx(1, 1)] + nibj_deriv[get_idx(0, 1)];
        derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0) += -nibj_deriv[get_idx(2, 0)] + nibj_deriv[get_idx(0, 0)];
    }
    if (hessian) {
        (*hessian)[1] = -nibj_hessian[get_idx(1, 1)] + nibj_hessian[get_idx(0, 1)];
        (*hessian)[1] += -nibj_hessian[get_idx(2, 0)] + nibj_hessian[get_idx(0, 0)];
    }

    // third entry: II(1, 0) = -(n1^T b1 - n0^T b1) - (n2^T b0 - n0^T b0) = II(0, 1)
    II(1, 0) = II(0, 1);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(2, 0) = derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0);
    }
    if (hessian) {
        (*hessian)[2] = (*hessian)[1];
    }

    // fourth entry: II(1, 1) = -2(n2^T b1 - n0^T b1)
    II(1, 1) = -2 * (nibj[get_idx(2, 1)] - nibj[get_idx(0, 1)]);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(3, 0) =
            -2 * (nibj_deriv[get_idx(2, 1)] - nibj_deriv[get_idx(0, 1)]);
    }
    if (hessian) {
        (*hessian)[3] = -2 * (nibj_hessian[get_idx(2, 1)] - nibj_hessian[get_idx(0, 1)]);
    }

    return II;
}

Eigen::Matrix2d MidedgeAngleGeneralSinFormulation::thirdFundamentalForm(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& extraDOFs,
    int face,
    Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs>* derivative,
    std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>>* hessian) {
    if (derivative) {
        derivative->setZero();
    }
    if (hessian) {
        hessian->resize(4);
        for (int i = 0; i < 4; i++) {
            (*hessian)[i].setZero();
        }
    }

    Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> n0n1_deriv, n0n2_deriv, n1n2_deriv;
    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> n0n1_hess, n0n2_hess, n1n2_hess;
    double n0n1 = compute_ninj(mesh, curPos, extraDOFs, face, 0, 1, derivative ? &n0n1_deriv : nullptr,
                        hessian ? &n0n1_hess : nullptr);
    double n0n2 = compute_ninj(mesh, curPos, extraDOFs, face, 0, 2, derivative ? &n0n2_deriv : nullptr,
                        hessian ? &n0n2_hess : nullptr);
    double n1n2 = compute_ninj(mesh, curPos, extraDOFs, face, 1, 2, derivative ? &n1n2_deriv : nullptr,
                        hessian ? &n1n2_hess : nullptr);

    Eigen::Matrix2d III;
    // first entry: III(0, 0) = 4(2 - 2 n0n1)
    III(0, 0) = 4 * (2 - 2 * n0n1);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(0, 0) =
            -8 * n0n1_deriv;
    }
    if (hessian) {
        (*hessian)[0] = -8 * n0n1_hess;
    }

    // second entry: II(0, 1) = 4 (1 + n1n2 - n0n2 - n0n1)
    III(0, 1) = 4 * (1 + n1n2 - n0n2 - n0n1);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0) =
            4 * (n1n2_deriv - n0n2_deriv - n0n1_deriv);
    }
    if (hessian) {
        (*hessian)[1] = 4 * (n1n2_hess - n0n2_hess - n0n1_hess);
    }

    // third entry: II(1, 0) = II(0, 1)
    III(1, 0) = III(0, 1);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(2, 0) = derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0);
    }
    if (hessian) {
        (*hessian)[2] = (*hessian)[1];
    }

    // fourth entry: III(1, 1) = 4 (2 - 2 n0n2)
    III(1, 1) = 4 * (2 - 2 * n0n2);
    if (derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(3, 0) =
            -8 * n0n2_deriv;
    }
    if (hessian) {
        (*hessian)[3] = -8 * n0n2_hess;
    }

    return III;
}

void MidedgeAngleGeneralSinFormulation::initializeExtraDOFs(Eigen::VectorXd& extraDOFs,
                                                            const MeshConnectivity& mesh,
                                                            const Eigen::MatrixXd& curPos) {
    int nedges = mesh.nEdges();
    extraDOFs.resize(numExtraDOFs * nedges);
    extraDOFs.setZero();

    for (int i = 0; i < nedges; i++) {
        extraDOFs[numExtraDOFs * i + 1] = M_PI_2;  // pi / 2, namely perpendicular to the edge
    }
}

std::vector<MidedgeAngleGeneralSinFormulation::VectorRelationship>
MidedgeAngleGeneralSinFormulation::get_edge_face_basis_relationship(const MeshConnectivity& mesh, int eid) {
    std::vector<VectorRelationship> basis_sign = {};
    Eigen::Matrix<VectorRelationship, 2, 3> face_basis_edge_relationship;
    face_basis_edge_relationship.row(0) << VectorRelationship::kPositiveOrientation,
        VectorRelationship::kNegativeOrientation, VectorRelationship::kSameDirection;
    face_basis_edge_relationship.row(1) << VectorRelationship::kPositiveOrientation,
        VectorRelationship::kOppositeDirection, VectorRelationship::kNegativeOrientation;

    for (int j = 0; j < 2; j++) {
        int fid = mesh.edgeFace(eid, j);

        if (fid == -1) {
            basis_sign.push_back(VectorRelationship::kUndefined);
            basis_sign.push_back(VectorRelationship::kUndefined);
            continue;
        }

        int feid = -1;
        for (int k = 0; k < 3; k++) {
            if (mesh.faceEdge(fid, k) == eid) {
                feid = k;
                break;
            }
        }
        assert(feid != -1);

        for (int k = 0; k < 2; k++) {
            VectorRelationship res = VectorRelationship::kUndefined;

            VectorRelationship edge_base_relationship = face_basis_edge_relationship(k, feid);
            if (mesh.faceEdgeOrientation(fid, feid) == 1) {
                if (edge_base_relationship == VectorRelationship::kSameDirection) {
                    res = VectorRelationship::kOppositeDirection;
                }
                if (edge_base_relationship == VectorRelationship::kOppositeDirection) {
                    res = VectorRelationship::kSameDirection;
                }
                if (edge_base_relationship == VectorRelationship::kPositiveOrientation) {
                    res = VectorRelationship::kNegativeOrientation;
                }
                if (edge_base_relationship == VectorRelationship::kNegativeOrientation) {
                    res = VectorRelationship::kPositiveOrientation;
                }
            } else {
                res = edge_base_relationship;
            }

            basis_sign.push_back(res);
        }
    }

    return basis_sign;
}

std::vector<Eigen::Vector3d> MidedgeAngleGeneralSinFormulation::get_face_edge_normals(const MeshConnectivity& mesh,
                                                                                      const Eigen::MatrixXd& curPos,
                                                                                      const Eigen::VectorXd& edgeDOFs,
                                                                                      int face) {
    // di = cos(σ) ê + sin(σ) cos(ζ) nf + sin(σ) sin(ζ) (ê x nf)
    // mi = 1
    Eigen::Vector3d b0 = curPos.row(mesh.faceVertex(face, 1)) - curPos.row(mesh.faceVertex(face, 0));
    Eigen::Vector3d b1 = curPos.row(mesh.faceVertex(face, 2)) - curPos.row(mesh.faceVertex(face, 0));
    Eigen::Vector3d nf = b0.cross(b1);
    nf.normalize();

    std::vector<Eigen::Vector3d> face_edge_normals = {};

    for (int i = 0; i < 3; i++) {
        int eid = mesh.faceEdge(face, i);

        Eigen::Vector3d e = curPos.row(mesh.edgeVertex(eid, 1)) - curPos.row(mesh.edgeVertex(eid, 0));
        e.normalize();

        Eigen::Vector3d eperp = e.cross(nf);

        double theta = edgeTheta(mesh, curPos, eid, nullptr, nullptr);

        double orient = mesh.faceEdgeOrientation(face, i) == 0 ? 1.0 : -1.0;
        double zeta = orient * 0.5 * theta + edgeDOFs[numExtraDOFs * eid];
        double sigma = edgeDOFs(eid * numExtraDOFs + 1);

        double mi = 1.0;
        Eigen::Vector3d di =
            std::cos(sigma) * e + std::sin(sigma) * std::cos(zeta) * nf + std::sin(sigma) * std::sin(zeta) * eperp;
        face_edge_normals.push_back(mi * di);
    }

    return face_edge_normals;
}

void MidedgeAngleGeneralSinFormulation::get_per_edge_face_sigma_zeta(const MeshConnectivity& mesh,
                                                                     const Eigen::MatrixXd& curPos,
                                                                     const Eigen::VectorXd& edgeDOFs,
                                                                     int edge,
                                                                     int face,
                                                                     double& sigma,
                                                                     double& zeta) {
    sigma = edgeDOFs(edge * numExtraDOFs + 1);
    zeta = 0;
    int face_id = mesh.edgeFace(edge, face);
    if (face_id == -1) {
        return;
    }
    double theta = edgeTheta(mesh, curPos, edge, nullptr, nullptr);

    double orient = face == 0 ? 1.0 : -1.0;
    zeta = orient * 0.5 * theta + edgeDOFs[numExtraDOFs * edge];
}

void MidedgeAngleGeneralSinFormulation::test_compute_nibj(const MeshConnectivity& mesh,
                                                          const Eigen::MatrixXd& curPos,
                                                          const Eigen::VectorXd& edgeDOFs,
                                                          int face,
                                                          int i,
                                                          int j) {
    auto to_variables = [&]() {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = curPos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = curPos.row(opp_vid);
            }
        }

        int eid = mesh.faceEdge(face, i);
        vars.segment<numExtraDOFs>(18 + numExtraDOFs * i) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }
        }
        int eid = mesh.faceEdge(face, i);
        edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * i);
    };

    Eigen::VectorXd vars = to_variables();
    Eigen::MatrixXd pos = curPos;
    Eigen::VectorXd edge_dofs = edgeDOFs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian,
                    bool is_proj) {
        from_variable(x, pos, edge_dofs);
        Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> dense_deriv;
        Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> dense_hess;
        double val = compute_nibj(mesh, pos, edge_dofs, face, i, j, deriv ? &dense_deriv : nullptr,
                                  hessian ? &dense_hess : nullptr);
        if (deriv) {
            *deriv = dense_deriv.transpose();
        }
        if (hessian) {
            std::vector<Eigen::Triplet<double>> T;
            for (int k = 0; k < dense_hess.rows(); k++) {
                for (int l = 0; l < dense_hess.cols(); l++) {
                    if (dense_hess(k, l) != 0) {
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

void MidedgeAngleGeneralSinFormulation::test_compute_ninj(const MeshConnectivity& mesh,
                                                          const Eigen::MatrixXd& curPos,
                                                          const Eigen::VectorXd& edgeDOFs,
                                                          int face,
                                                          int i,
                                                          int j) {
    std::vector<Eigen::Vector3d> face_edge_normals = get_face_edge_normals(mesh, curPos, edgeDOFs, face);
    Eigen::Vector3d ni = face_edge_normals[i];
    Eigen::Vector3d nj = face_edge_normals[j];
    double ninj = ni.dot(nj);
    double ninj_computed = compute_ninj(mesh, curPos, edgeDOFs, face, i, j, nullptr, nullptr);

    double diff = std::abs(ninj - ninj_computed);
    std::cout << "Difference from different computation: " << std::abs(ninj - ninj_computed) << std::endl;

    if (diff > 1e-10) {
        std::cout << "Error: ninj mismatch!" << std::endl;
        std::cout << "ninj: " << ninj << ", ninj_computed: " << ninj_computed << std::endl;
    }

    auto to_variables = [&]() {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = curPos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = curPos.row(opp_vid);
            }
        }

        int eid = mesh.faceEdge(face, i);
        vars.segment<numExtraDOFs>(18 + numExtraDOFs * i) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);

        eid = mesh.faceEdge(face, j);
        vars.segment<numExtraDOFs>(18 + numExtraDOFs * j) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }
        }
        int eid = mesh.faceEdge(face, i);
        edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * i);

        int eid2 = mesh.faceEdge(face, j);
        edge_dofs.segment<numExtraDOFs>(eid2 * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * j);
    };

    Eigen::VectorXd vars = to_variables();
    Eigen::MatrixXd pos = curPos;
    Eigen::VectorXd edge_dofs = edgeDOFs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian,
                    bool is_proj) {
        from_variable(x, pos, edge_dofs);
        Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> dense_deriv;
        Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> dense_hess;
        double val = compute_ninj(mesh, pos, edge_dofs, face, i, j, deriv ? &dense_deriv : nullptr,
                                  hessian ? &dense_hess : nullptr);
        if (deriv) {
            *deriv = dense_deriv.transpose();
        }
        if (hessian) {
            std::vector<Eigen::Triplet<double>> T;
            for (int k = 0; k < dense_hess.rows(); k++) {
                for (int l = 0; l < dense_hess.cols(); l++) {
                    if (dense_hess(k, l) != 0) {
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

void MidedgeAngleGeneralSinFormulation::test_second_fund_form(const MeshConnectivity& mesh,
                                                              const Eigen::MatrixXd& curPos,
                                                              const Eigen::VectorXd& edgeDOFs,
                                                              int face) {
    auto to_variables = [&]() {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = curPos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = curPos.row(opp_vid);
            }

            int eid = mesh.faceEdge(face, k);
            vars.segment<numExtraDOFs>(18 + numExtraDOFs * k) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }

            int eid = mesh.faceEdge(face, k);
            edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * k);
        }
    };

    Eigen::VectorXd vars = to_variables();
    Eigen::MatrixXd pos = curPos;
    Eigen::VectorXd edge_dofs = edgeDOFs;

    int selected_entry = 0;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian,
                    bool is_proj) {
        from_variable(x, pos, edge_dofs);
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs> dense_deriv;
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>> dense_hess;
        Eigen::Matrix2d II = secondFundamentalForm(mesh, pos, edge_dofs, face, deriv ? &dense_deriv : nullptr,
                                                   hessian ? &dense_hess : nullptr);
        if (deriv) {
            *deriv = dense_deriv.row(selected_entry).transpose();
        }
        if (hessian) {
            std::vector<Eigen::Triplet<double>> T;
            for (int k = 0; k < dense_hess[selected_entry].rows(); k++) {
                for (int l = 0; l < dense_hess[selected_entry].cols(); l++) {
                    if (dense_hess[selected_entry](k, l) != 0) {
                        T.push_back(Eigen::Triplet<double>(k, l, dense_hess[selected_entry](k, l)));
                    }
                }
            }
            hessian->resize(18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        int row = selected_entry / 2;
        int col = selected_entry % 2;
        return II(row, col);
    };
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << "======================= test (" << i << ", " << j
                      << ") entry =======================" << std::endl;
            selected_entry = i * 2 + j;
            TestFuncGradHessian(func, vars);
        }
    }
}

void MidedgeAngleGeneralSinFormulation::test_third_fund_form(const MeshConnectivity& mesh,
                                                              const Eigen::MatrixXd& curPos,
                                                              const Eigen::VectorXd& edgeDOFs,
                                                              int face) {
    std::vector<Eigen::Vector3d> face_edge_normals = get_face_edge_normals(mesh, curPos, edgeDOFs, face);
    auto& n0 = face_edge_normals[0];
    auto& n1 = face_edge_normals[1];
    auto& n2 = face_edge_normals[2];

    Eigen::Matrix2d direct_III;
    direct_III << (n1 - n0).dot(n1 - n0), (n1 - n0).dot(n2 - n0),
        (n2 - n0).dot(n1 - n0), (n2 - n0).dot(n2 - n0);
    direct_III *= 4;

    Eigen::Matrix2d III = thirdFundamentalForm(mesh, curPos, edgeDOFs, face, nullptr, nullptr);

    std::cout << "======================= test III =======================" << std::endl;
    std::cout << "Result difference from direct computation: " << (direct_III - III).norm() << std::endl;

    auto to_variables = [&]() {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = curPos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = curPos.row(opp_vid);
            }

            int eid = mesh.faceEdge(face, k);
            vars.segment<numExtraDOFs>(18 + numExtraDOFs * k) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for (int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            pos.row(vid) = vars.segment<3>(3 * k);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if (opp_vid != -1) {
                pos.row(opp_vid) = vars.segment<3>(9 + 3 * k);
            }

            int eid = mesh.faceEdge(face, k);
            edge_dofs.segment<numExtraDOFs>(eid * numExtraDOFs) = vars.segment<numExtraDOFs>(18 + numExtraDOFs * k);
        }
    };

    Eigen::VectorXd vars = to_variables();
    Eigen::MatrixXd pos = curPos;
    Eigen::VectorXd edge_dofs = edgeDOFs;

    int selected_entry = 0;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian,
                    bool is_proj) {
        from_variable(x, pos, edge_dofs);
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs> dense_deriv;
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>> dense_hess;
        Eigen::Matrix2d III = thirdFundamentalForm(mesh, pos, edge_dofs, face, deriv ? &dense_deriv : nullptr,
                                                   hessian ? &dense_hess : nullptr);
        if (deriv) {
            *deriv = dense_deriv.row(selected_entry).transpose();
        }
        if (hessian) {
            std::vector<Eigen::Triplet<double>> T;
            for (int k = 0; k < dense_hess[selected_entry].rows(); k++) {
                for (int l = 0; l < dense_hess[selected_entry].cols(); l++) {
                    if (dense_hess[selected_entry](k, l) != 0) {
                        T.push_back(Eigen::Triplet<double>(k, l, dense_hess[selected_entry](k, l)));
                    }
                }
            }
            hessian->resize(18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        int row = selected_entry / 2;
        int col = selected_entry % 2;
        return III(row, col);
    };
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << "======================= test (" << i << ", " << j
                      << ") entry =======================" << std::endl;
            selected_entry = i * 2 + j;
            TestFuncGradHessian(func, vars);
        }
    }
}

};  // namespace LibShell