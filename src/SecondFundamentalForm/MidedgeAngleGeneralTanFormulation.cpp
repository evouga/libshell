//
// Created by Zhen Chen on 11/25/24.
//
#include "../../include/MidedgeAngleGeneralTanFormulation.h"
#include "../../include/MeshConnectivity.h"
#include "../GeometryDerivatives.h"

#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <iostream>
#include <random>

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

// Define the static member variable
std::vector<std::array<MidedgeAngleGeneralTanFormulation::VectorRelationship, 2>>
    MidedgeAngleGeneralTanFormulation::m_edge_face_basis_sign;
constexpr int MidedgeAngleGeneralTanFormulation::numExtraDOFs;

// ni^T bj = cot(σ) / cos(ζ) (bj^T ei) / |ei| - tan(ζ) s_ij * |bj x ei| / |ei|,
//         = cot(σ) / cos(ζ) (bj^T ei) / |ei| - tan(ζ) s_ij * h_i (if bj is not parallel to ei),
//         = cot(σ) / cos(ζ) |ei| * sign(bj^T ei) (if bj is parallel to ei),
double MidedgeAngleGeneralTanFormulation::compute_nibj(const MeshConnectivity& mesh,
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

    VectorRelationship& vector_relationship = m_edge_face_basis_sign[2 * eid + efid][j];
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

    Eigen::Matrix<double, 1, 12> thetaderiv;
    Eigen::Matrix<double, 12, 12> thetahess;
    double theta =
        edgeTheta(mesh, curPos, eid, (derivative || hessian) ? &thetaderiv : nullptr, hessian ? &thetahess : nullptr);

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

    double orient = mesh.faceEdgeOrientation(face, i) == 0 ? 1.0 : -1.0;
    double zeta = orient * 0.5 * theta + edgeDOFs[numExtraDOFs * eid];

    double tan_cos = std::tan(sigma) * std::cos(zeta);

    Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> tan_cos_deriv;
    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> tan_cos_hess;

    if(derivative || hessian) {
        tan_cos_deriv.setZero();
        for (int k = 0; k < 4; k++) {
            tan_cos_deriv.block<1, 3>(0, 3 * av[k]) +=
                -orient * 0.5 * std::tan(sigma) * std::sin(zeta) * thetaderiv.block<1, 3>(0, 3 * k);
        }

        tan_cos_deriv(18 + i * numExtraDOFs) = -std::tan(sigma) * std::sin(zeta);
        tan_cos_deriv(18 + i * numExtraDOFs + 1) = std::cos(zeta) / (std::cos(sigma) * std::cos(sigma));

        if(hessian) {
            tan_cos_hess.setZero();
            tan_cos_hess(18 + i * numExtraDOFs, 18 + i * numExtraDOFs) += -std::tan(sigma) * std::cos(zeta);
            tan_cos_hess(18 + i * numExtraDOFs, 18 + i * numExtraDOFs + 1) += -std::sin(zeta) / (std::cos(sigma) * std::cos(sigma));
            for(int k = 0; k < 4; k++) {
                tan_cos_hess.block<1, 3>(18 + i * numExtraDOFs, 3 * av[k]) += -std::tan(sigma) * std::cos(zeta) * 0.5 * orient * thetaderiv.block<1, 3>(0, 3 * k);
            }

            tan_cos_hess(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs) += -std::sin(zeta) / (std::cos(sigma) * std::cos(sigma));
            tan_cos_hess(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) += 2 / (std::cos(sigma) * std::cos(sigma)) * std::tan(sigma) * std::cos(zeta);
            for(int k = 0; k < 4; k++) {
                tan_cos_hess.block<1, 3>(18 + i * numExtraDOFs + 1, 3 * av[k]) += -std::sin(zeta) / (std::cos(sigma) * std::cos(sigma)) * orient * 0.5 * thetaderiv.block<1, 3>(0, 3 * k);
            }

            for(int k = 0; k < 4; k++) {
                for(int m = 0; m < 4; m++) {
                    tan_cos_hess.block<3, 3>(3 * av[k], 3 * av[m]) += -std::tan(sigma) * std::sin(zeta) * orient * 0.5 * thetahess.block<3, 3>(3 * k, 3 * m);
                    tan_cos_hess.block<3, 3>(3 * av[k], 3 * av[m]) += -std::tan(sigma) * std::cos(zeta) * 0.25 * thetaderiv.block<1, 3>(0, 3 * k).transpose() * thetaderiv.block<1, 3>(0, 3 * m);
                }
                tan_cos_hess.block<3, 1>(3 * av[k], 18 + i * numExtraDOFs) += -orient * 0.5 * std::tan(sigma) * std::cos(zeta) * thetaderiv.block<1, 3>(0, 3 * k).transpose();

                tan_cos_hess.block<3, 1>(3 * av[k], 18 + i * numExtraDOFs + 1) += -std::sin(zeta) / (std::cos(sigma) * std::cos(sigma)) * orient * 0.5 * thetaderiv.block<1, 3>(0, 3 * k).transpose();
            }

        }
    }

    // ni^T bj = cot(σ) / cos(ζ) |ei| * sign(bj^T ei)
    if (vector_relationship == VectorRelationship::kSameDirection ||
        vector_relationship == VectorRelationship::kOppositeDirection) {
        double sign = vector_relationship == VectorRelationship::kSameDirection ? 1.0 : -1.0;
        double res = enorm / tan_cos * sign;


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
        // ni^T bj = mi cos(sigma) bj^T ei / |ei| - mi sin(sigma) sin(zeta) sij * hi, if bj is not parallel to ei
        Eigen::Vector3d bj = curPos.row(mesh.faceVertex(face, (j + 1) % 3)) - curPos.row(mesh.faceVertex(face, 0));
        double dot_prod = bj.dot(e);
        double dot_over_norm = dot_prod / enorm;
        double part1 = std::cos(sigma) * dot_over_norm;

        Eigen::Matrix<double, 1, 9> hderiv;
        Eigen::Matrix<double, 9, 9> hhess;
        double altitude =
            triangleAltitude(mesh, curPos, face, i, (derivative || hessian) ? &hderiv : nullptr, hessian ? &hhess : nullptr);
        

        double sij = vector_relationship == VectorRelationship::kPositiveOrientation ? 1.0 : -1.0;
        double part2 = -sij * std::sin(sigma) * std::sin(zeta) * altitude;

        if (derivative || hessian) {
            // derivatives and hessian from first part
            {
                // mi cos(sigma) bj^T ei / |ei|
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
                        2 * dot_prod * norm_deriv_full.transpose() * norm_deriv_full /
                            (enorm * enorm * enorm);

                    // hessian->block<9, 9>(0, 0) += norm_hess_full;

                    (*hessian)(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) +=
                        -dot_over_norm * std::cos(sigma);
                    hessian->block<1, 9>(18 + i * numExtraDOFs + 1, 0) += -std::sin(sigma) * dot_over_norm_deriv;

                    hessian->block<9, 9>(0, 0) += std::cos(sigma) * dot_over_norm_hess;
                    hessian->block<9, 1>(0, 18 + i * numExtraDOFs + 1) +=
                        -std::sin(sigma) * dot_over_norm_deriv.transpose();
                }
            }

            // derivatives and hessian from second part
            {
                int hv[3];
                for (int k = 0; k < 3; k++) {
                    hv[k] = (i + k) % 3;
                }

                // -sij * std::sin(sigma) * std::sin(zeta) * altitude;
                double sin_zeta_altitude = std::sin(zeta) * altitude;
                Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> sin_zeta_altitude_deriv;
                sin_zeta_altitude_deriv.setZero();
                
                for (int k = 0; k < 3; k++) {
                    sin_zeta_altitude_deriv.block<1, 3>(0, 3 * hv[k]) += sin(zeta) * hderiv.block<1, 3>(0, 3 * k);
                }

                for (int k = 0; k < 4; k++) {
                    sin_zeta_altitude_deriv.block<1, 3>(0, 3 * av[k]) +=
                        orient * 0.5 * altitude * cos(zeta) * thetaderiv.block<1, 3>(0, 3 * k);
                }
                sin_zeta_altitude_deriv(0, 18 + i * numExtraDOFs) += altitude * cos(zeta);

                if(derivative) {
                    (*derivative) += -sij * std::sin(sigma) * sin_zeta_altitude_deriv;
                    (*derivative)(0, 18 + i * numExtraDOFs + 1) += -sij * std::cos(sigma) * sin_zeta_altitude;
                }
                
                if(hessian) {
                    Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> sin_altitude_hess;
                    sin_altitude_hess.setZero();

                    for (int m = 0; m < 3; m++)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            sin_altitude_hess.block<3, 3>(3 * hv[m], 3 * hv[k]) += sin(zeta) * hhess.block<3, 3>(3 * m, 3 * k);
                        }
                    }

                    for (int k = 0; k < 3; k++)
                    {
                        for (int m = 0; m < 4; m++)
                        {
                            sin_altitude_hess.block<3, 3>(3 * av[m], 3 * hv[k]) += orient * 0.5 * cos(zeta) * thetaderiv.block(0, 3 * m, 1, 3).transpose() * hderiv.block(0, 3 * k, 1, 3);
                            sin_altitude_hess.block<3, 3>(3 * hv[k], 3 * av[m]) += orient * 0.5 * cos(zeta) * hderiv.block(0, 3 * k, 1, 3).transpose() * thetaderiv.block(0, 3 * m, 1, 3);
                        }
                        sin_altitude_hess.block<1, 3>(18 + i * numExtraDOFs, 3 * hv[k]) += cos(zeta) * hderiv.block<1, 3>(0, 3 * k);
                        sin_altitude_hess.block<3, 1>(3 * hv[k], 18 + i * numExtraDOFs) += cos(zeta) * hderiv.block<1, 3>(0, 3 * k).transpose();
                    }

                    for (int k = 0; k < 4; k++)
                    {
                        for (int m = 0; m < 4; m++)
                        {
                            sin_altitude_hess.block<3, 3>(3 * av[m], 3 * av[k]) += orient * 0.5 * altitude * cos(zeta) * thetahess.block<3, 3>(3 * m, 3 * k);
                            sin_altitude_hess.block<3, 3>(3 * av[m], 3 * av[k]) += -0.25 * altitude * sin(zeta) * thetaderiv.block<1, 3>(0, 3 * m).transpose() * thetaderiv.block<1, 3>(0, 3 * k);
                        }
                        sin_altitude_hess.block<1, 3>(18 + i * numExtraDOFs, 3 * av[k]) += -0.5 * altitude * sin(zeta) * orient * thetaderiv.block<1, 3>(0, 3 * k);
                        sin_altitude_hess.block<3, 1>(3 * av[k], 18 + i * numExtraDOFs) += -0.5 * altitude * sin(zeta) * orient * thetaderiv.block<1, 3>(0, 3 * k).transpose();
                    }

                    sin_altitude_hess(18 + i * numExtraDOFs, 18 + i * numExtraDOFs) += -altitude * sin(zeta);

                    (*hessian)(18 + i * numExtraDOFs + 1, 18 + i * numExtraDOFs + 1) += sij * std::sin(sigma) * sin_zeta_altitude;
                    hessian->block<1, 18 + 3 * numExtraDOFs>(18 + i * numExtraDOFs + 1, 0) += -sij * std::cos(sigma) * sin_zeta_altitude_deriv;

                    (*hessian) += -sij * std::sin(sigma) * sin_altitude_hess;
                    hessian->block<18 + 3 * numExtraDOFs, 1>(0, 18 + i * numExtraDOFs + 1) += -sij * std::cos(sigma) * sin_zeta_altitude_deriv.transpose();

                }
            }
        }

        return part1 + part2;
    }
}

Eigen::Matrix2d MidedgeAngleGeneralTanFormulation::secondFundamentalForm(
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

    for(int i = 0; i < 6; i++) {
        int edge = i / 2;
        int bid = i % 2;
        nibj[i] = compute_nibj(mesh, curPos, extraDOFs, face, edge, bid, derivative ? &nibj_deriv[i] : nullptr, hessian ? &nibj_hessian[i] : nullptr);
    }

    auto get_idx = [&](int i, int j) -> int {
        return i * 2 + j;
    };

    Eigen::Matrix2d II;
    // first entry: II(0, 0) = 2(n1^T b0 - n0^T b0)
    II(0, 0) = 2 * (nibj[get_idx(1, 0)] - nibj[get_idx(0, 0)]);
    if(derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(0, 0) = 2 * (nibj_deriv[get_idx(1, 0)] - nibj_deriv[get_idx(0, 0)]);
    }
    if(hessian) {
        (*hessian)[0] = 2 * (nibj_hessian[get_idx(1, 0)] - nibj_hessian[get_idx(0, 0)]);
    }


    // second entry: II(0, 1) = (n1^T b1 - n0^T b1) + (n2^T b0 - n0^T b0)
    II(0, 1) = nibj[get_idx(1, 1)] - nibj[get_idx(0, 1)] + nibj[get_idx(2, 0)] - nibj[get_idx(0, 0)];
    if(derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0) = nibj_deriv[get_idx(1, 1)] - nibj_deriv[get_idx(0, 1)];
        derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0) += nibj_deriv[get_idx(2, 0)] - nibj_deriv[get_idx(0, 0)];
    }
    if(hessian) {
        (*hessian)[1] = nibj_hessian[get_idx(1, 1)] - nibj_hessian[get_idx(0, 1)];
        (*hessian)[1] += nibj_hessian[get_idx(2, 0)] - nibj_hessian[get_idx(0, 0)];
    }

    // third entry: II(1, 0) = (n1^T b1 - n0^T b1) + (n2^T b0 - n0^T b0) = II(0, 1)
    II(1, 0) = II(0, 1);
    if(derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(2, 0) = derivative->block<1, 18 + 3 * numExtraDOFs>(1, 0);
    }
    if(hessian) {
        (*hessian)[2] = (*hessian)[1];
    }

    // fourth entry: II(1, 1) = 2(n2^T b1 - n0^T b1)
    II(1, 1) = 2 * (nibj[get_idx(2, 1)] - nibj[get_idx(0, 1)]);
    if(derivative) {
        derivative->block<1, 18 + 3 * numExtraDOFs>(3, 0) = 2 * (nibj_deriv[get_idx(2, 1)] - nibj_deriv[get_idx(0, 1)]);
    }
    if(hessian) {
        (*hessian)[3] = 2 * (nibj_hessian[get_idx(2, 1)] - nibj_hessian[get_idx(0, 1)]);
    }

    return II;
}

void MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(Eigen::VectorXd& extraDOFs,
                                                         const MeshConnectivity& mesh,
                                                         const Eigen::MatrixXd& curPos) {
    int nedges = mesh.nEdges();
    extraDOFs.resize(numExtraDOFs * nedges);
    extraDOFs.setZero();
    m_edge_face_basis_sign.clear();

    for (int i = 0; i < nedges; i++) {
        extraDOFs[numExtraDOFs * i + 1] = M_PI_2;   // pi / 2, namely perpendicular to the edge

        Eigen::Vector3d e = curPos.row(mesh.edgeVertex(i, 1)) - curPos.row(mesh.edgeVertex(i, 0));
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            std::array<VectorRelationship, 2> basis_sign = {VectorRelationship::kUndefined,
                                                            VectorRelationship::kUndefined};
            if (fid == -1) {
                m_edge_face_basis_sign.push_back(basis_sign);
                continue;
            }

            std::vector<Eigen::Vector3d> bks(2);
            for (int k = 1; k <= 2; k++) {
                bks[k - 1] = curPos.row(mesh.faceVertex(fid, k)) - curPos.row(mesh.faceVertex(fid, 0));
            }
            Eigen::Vector3d face_normal = bks[0].cross(bks[1]);

            // ToDo: Add sanity check for really tiny faces
            if (face_normal.norm()) {
                face_normal.normalize();
            }

            for (int k = 0; k < 2; k++) {
                std::array<int, 2> vids = {mesh.faceVertex(fid, 0), mesh.faceVertex(fid, k + 1)};

                if (vids[0] == mesh.edgeVertex(i, 0) && vids[1] == mesh.edgeVertex(i, 1)) {
                    basis_sign[k] = VectorRelationship::kSameDirection;
                } else if (vids[0] == mesh.edgeVertex(i, 1) && vids[1] == mesh.edgeVertex(i, 0)) {
                    basis_sign[k] = VectorRelationship::kOppositeDirection;
                } else {
                    Eigen::Vector3d vk = bks[k].cross(e);
                    basis_sign[k] = (vk.dot(face_normal) > 0) ? VectorRelationship::kPositiveOrientation
                                                              : VectorRelationship::kNegativeOrientation;
                }
            }

            m_edge_face_basis_sign.push_back(basis_sign);
        }
    }
}


void MidedgeAngleGeneralTanFormulation::test_compute_nibj(const MeshConnectivity& mesh,
                                                       const Eigen::MatrixXd& curPos,
                                                       const Eigen::VectorXd& edgeDOFs,
                                                       int face,
                                                       int i,
                                                       int j) {
    auto to_variables = [&]() {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = curPos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if(opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = curPos.row(opp_vid);
            }
        }

        int eid = mesh.faceEdge(face, i);
        vars.segment<numExtraDOFs>(18 + numExtraDOFs * i) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for(int k = 0; k < 3; k++) {
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

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, pos, edge_dofs);
        Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs> dense_deriv;
        Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> dense_hess;
        double val = compute_nibj(mesh, pos, edge_dofs, face, i, j, deriv ? &dense_deriv : nullptr, hessian ? &dense_hess : nullptr);
        if(deriv) {
            *deriv = dense_deriv.transpose();
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

void MidedgeAngleGeneralTanFormulation::test_second_fund_form(const MeshConnectivity& mesh,
                                                       const Eigen::MatrixXd& curPos,
                                                       const Eigen::VectorXd& edgeDOFs,
                                                       int face) {
    auto to_variables = [&]() {
        Eigen::VectorXd vars(18 + 3 * numExtraDOFs);
        vars.setZero();
        for(int k = 0; k < 3; k++) {
            int vid = mesh.faceVertex(face, k);
            vars.segment<3>(3 * k) = curPos.row(vid);

            int opp_vid = mesh.vertexOppositeFaceEdge(face, k);
            if(opp_vid != -1) {
                vars.segment<3>(9 + 3 * k) = curPos.row(opp_vid);
            }

            int eid = mesh.faceEdge(face, k);
            vars.segment<numExtraDOFs>(18 + numExtraDOFs * k) = edgeDOFs.segment<numExtraDOFs>(eid * numExtraDOFs);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& edge_dofs) {
        assert(vars.size() == 18 + 3 * numExtraDOFs);
        for(int k = 0; k < 3; k++) {
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

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian, bool is_proj) {
        from_variable(x, pos, edge_dofs);
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs> dense_deriv;
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>> dense_hess;
        Eigen::Matrix2d II = secondFundamentalForm(mesh, pos, edge_dofs, face, deriv ? &dense_deriv : nullptr, hessian ? &dense_hess : nullptr);
        if(deriv) {
            *deriv = dense_deriv.row(selected_entry).transpose();
        }
        if(hessian) {
            std::vector<Eigen::Triplet<double>> T;
            for(int k = 0; k < dense_hess[selected_entry].rows(); k++) {
                for(int l = 0; l < dense_hess[selected_entry].cols(); l++) {
                    if(dense_hess[selected_entry](k, l) != 0) {
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
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            std::cout << "======================= test (" << i << ", " << j << ") entry =======================" << std::endl;
            selected_entry = i * 2 + j;
            TestFuncGradHessian(func, vars);
        }
    }
}
};  // namespace LibShell