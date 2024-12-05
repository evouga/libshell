#include "../../include/MidedgeAngleCompressiveFormulation.h"
#include "../../include/MeshConnectivity.h"
#include "../GeometryDerivatives.h"

#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <Eigen/Geometry>

namespace LibShell {
static Eigen::Vector3d secondFundamentalFormEntries(const MeshConnectivity& mesh,
                                                    const Eigen::MatrixXd& curPos,
                                                    const Eigen::VectorXd& edgeDOFs,
                                                    int face,
                                                    Eigen::Matrix<double, 3, 27>* derivative,
                                                    std::vector<Eigen::Matrix<double, 27, 27>>* hessian) {
    if (derivative) derivative->setZero();
    if (hessian) {
        hessian->resize(3);
        for (int i = 0; i < 3; i++) (*hessian)[i].setZero();
    }

    Eigen::Vector3d II;
    for (int i = 0; i < 3; i++) {
        Eigen::Matrix<double, 1, 9> hderiv;
        Eigen::Matrix<double, 9, 9> hhess;
        double altitude =
            triangleAltitude(mesh, curPos, face, i, (derivative || hessian) ? &hderiv : NULL, hessian ? &hhess : NULL);

        int edge = mesh.faceEdge(face, i);
        Eigen::Matrix<double, 1, 12> thetaderiv;
        Eigen::Matrix<double, 12, 12> thetahess;
        double theta =
            edgeTheta(mesh, curPos, edge, (derivative || hessian) ? &thetaderiv : NULL, hessian ? &thetahess : NULL);

        double orient = mesh.faceEdgeOrientation(face, i) == 0 ? 1.0 : -1.0;
        double alpha = 0.5 * theta + orient * edgeDOFs[3 * edge];

        int offset = mesh.faceEdgeOrientation(face, i) + 1;
        double vec_norm = edgeDOFs[3 * edge + offset];
        II[i] = 2.0 * altitude * tan(alpha) * vec_norm;

        if (derivative || hessian) {
            int hv[3];
            for (int j = 0; j < 3; j++) {
                hv[j] = (i + j) % 3;
            }

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

            if (derivative) {
                for(int j = 0; j < 3; j++) {
                    derivative->block(i, 3 * hv[j], 1, 3) += 2.0 * tan(alpha) * hderiv.block(0, 3 * j, 1, 3) * vec_norm;
                }
                for(int j = 0; j < 4; j++) {
                    derivative->block(i, 3 * av[j], 1, 3) +=
                    altitude / cos(alpha) / cos(alpha) * thetaderiv.block(0, 3 * j, 1, 3) * vec_norm;
                }
                (*derivative)(i, 18 + 3 * i) += 2.0 * altitude / cos(alpha) / cos(alpha) * orient * vec_norm;
                (*derivative)(i, 18 + 3 * i + offset) += 2.0 * tan(alpha) * altitude;
            }

            if (hessian) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        (*hessian)[i].block(3 * hv[j], 3 * hv[k], 3, 3) +=
                            2.0 * tan(alpha) * hhess.block(3 * j, 3 * k, 3, 3) * vec_norm;
                    }
                }

                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < 4; j++) {
                        (*hessian)[i].block(3 * av[j], 3 * hv[k], 3, 3) +=
                            1.0 / cos(alpha) / cos(alpha) * thetaderiv.block(0, 3 * j, 1, 3).transpose() *
                            hderiv.block(0, 3 * k, 1, 3) * vec_norm;
                        (*hessian)[i].block(3 * hv[k], 3 * av[j], 3, 3) += 1.0 / cos(alpha) / cos(alpha) *
                                                                           hderiv.block(0, 3 * k, 1, 3).transpose() *
                                                                           thetaderiv.block(0, 3 * j, 1, 3) * vec_norm;
                    }
                    (*hessian)[i].block(18 + 3 * i, 3 * hv[k], 1, 3) +=
                        2.0 / cos(alpha) / cos(alpha) * orient * hderiv.block(0, 3 * k, 1, 3) * vec_norm;
                    (*hessian)[i].block(3 * hv[k], 18 + 3 * i, 3, 1) +=
                        2.0 / cos(alpha) / cos(alpha) * orient * hderiv.block(0, 3 * k, 1, 3).transpose() * vec_norm;
                }

                for (int k = 0; k < 4; k++) {
                    for (int j = 0; j < 4; j++) {
                        (*hessian)[i].block(3 * av[j], 3 * av[k], 3, 3) +=
                            altitude / cos(alpha) / cos(alpha) * thetahess.block(3 * j, 3 * k, 3, 3) * vec_norm;
                        (*hessian)[i].block(3 * av[j], 3 * av[k], 3, 3) +=
                            altitude * tan(alpha) / cos(alpha) / cos(alpha) *
                            thetaderiv.block(0, 3 * j, 1, 3).transpose() * thetaderiv.block(0, 3 * k, 1, 3) * vec_norm;
                    }
                    (*hessian)[i].block(18 + 3 * i, 3 * av[k], 1, 3) += 2.0 * altitude * tan(alpha) / cos(alpha) /
                                                                        cos(alpha) * orient *
                                                                        thetaderiv.block(0, 3 * k, 1, 3) * vec_norm;
                    (*hessian)[i].block(3 * av[k], 18 + 3 * i, 3, 1) +=
                        2.0 * altitude * tan(alpha) / cos(alpha) / cos(alpha) * orient *
                        thetaderiv.block(0, 3 * k, 1, 3).transpose() * vec_norm;
                }

                (*hessian)[i](18 + 3 * i, 18 + 3 * i) += 4.0 * altitude * tan(alpha) / cos(alpha) / cos(alpha) * vec_norm;

                // extra norm dofs
                for (int j = 0; j < 3; j++) {
                    (*hessian)[i].block(18 + 3 * i + offset, 3 * hv[j], 1, 3) +=
                        2.0 * tan(alpha) * hderiv.block(0, 3 * j, 1, 3);
                    (*hessian)[i].block(3 * hv[j], 18 + 3 * i + offset, 3, 1) +=
                        2.0 * tan(alpha) * hderiv.block(0, 3 * j, 1, 3).transpose();
                }

                for (int j = 0; j < 4; j++) {
                    (*hessian)[i].block(18 + 3 * i + offset, 3 * av[j], 1, 3) +=
                        altitude / cos(alpha) / cos(alpha) * thetaderiv.block(0, 3 * j, 1, 3);
                    (*hessian)[i].block(3 * av[j], 18 + 3 * i + offset, 3, 1) +=
                        altitude / cos(alpha) / cos(alpha) * thetaderiv.block(0, 3 * j, 1, 3).transpose();
                }

                (*hessian)[i](18 + 3 * i + offset, 18 + 3 * i) +=
                    2.0 * altitude / cos(alpha) / cos(alpha) * orient;
                (*hessian)[i](18 + 3 * i, 18 + 3 * i + offset) += 2.0 * altitude / cos(alpha) / cos(alpha) * orient;
            }
        }
    }

    return II;
}

Eigen::Matrix2d MidedgeAngleCompressiveFormulation::secondFundamentalForm(
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

    Eigen::Matrix<double, 3, 27> IIderiv;
    std::vector<Eigen::Matrix<double, 27, 27>> IIhess;

    Eigen::Vector3d II = secondFundamentalFormEntries(mesh, curPos, extraDOFs, face, derivative ? &IIderiv : NULL,
                                                      hessian ? &IIhess : NULL);
    Eigen::Matrix2d result;
    result << II[0] + II[1], II[0], II[0], II[0] + II[2];

    if (derivative) {
        derivative->row(0) += IIderiv.row(0);
        derivative->row(0) += IIderiv.row(1);

        derivative->row(1) += IIderiv.row(0);
        derivative->row(2) += IIderiv.row(0);

        derivative->row(3) += IIderiv.row(0);
        derivative->row(3) += IIderiv.row(2);
    }
    if (hessian) {
        (*hessian)[0] += IIhess[0];
        (*hessian)[0] += IIhess[1];

        (*hessian)[1] += IIhess[0];
        (*hessian)[2] += IIhess[0];

        (*hessian)[3] += IIhess[0];
        (*hessian)[3] += IIhess[2];
    }

    return result;
}

constexpr int MidedgeAngleCompressiveFormulation::numExtraDOFs;

void MidedgeAngleCompressiveFormulation::initializeExtraDOFs(Eigen::VectorXd& extraDOFs,
                                                             const MeshConnectivity& mesh,
                                                             const Eigen::MatrixXd& curPos) {
    extraDOFs.resize(numExtraDOFs * mesh.nEdges());
    extraDOFs.setZero();
    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 1; j < numExtraDOFs; j++) {
            extraDOFs[3 * i + j] = 1;
        }
    }
}

};  // namespace LibShell