#include "GeometryDerivatives.h"
#include "../include/MeshConnectivity.h"
#include <iostream>
#include <random>
#include <Eigen/Geometry>

namespace LibShell {

Eigen::Matrix3d crossMatrix(Eigen::Vector3d v) {
    Eigen::Matrix3d ret;
    ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return ret;
}

Eigen::Matrix2d adjugate(Eigen::Matrix2d M) {
    Eigen::Matrix2d ret;
    ret << M(1, 1), -M(0, 1), -M(1, 0), M(0, 0);
    return ret;
}

double angle(const Eigen::Vector3d& v,
             const Eigen::Vector3d& w,
             const Eigen::Vector3d& axis,
             Eigen::Matrix<double, 1, 9>* derivative,  // v, w
             Eigen::Matrix<double, 9, 9>* hessian) {
    // sanity check
    if(!v.norm() || !w.norm() || !axis.norm()) {
        if(derivative) {
            derivative->setZero();
        }
        if(hessian) {
            hessian->setZero();
        }
        return 0;
    }

    double theta = 2.0 * atan2((v.cross(w).dot(axis) / axis.norm()), v.dot(w) + v.norm() * w.norm());

    if (derivative) {
        derivative->segment<3>(0) = -axis.cross(v) / v.squaredNorm() / axis.norm();
        derivative->segment<3>(3) = axis.cross(w) / w.squaredNorm() / axis.norm();
        derivative->segment<3>(6).setZero();
    }
    if (hessian) {
        hessian->setZero();
        hessian->block<3, 3>(0, 0) +=
            2.0 * (axis.cross(v)) * v.transpose() / v.squaredNorm() / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) +=
            -2.0 * (axis.cross(w)) * w.transpose() / w.squaredNorm() / w.squaredNorm() / axis.norm();
        hessian->block<3, 3>(0, 0) += -crossMatrix(axis) / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) += crossMatrix(axis) / w.squaredNorm() / axis.norm();

        Eigen::Matrix3d dahat = (Eigen::Matrix3d::Identity() / axis.norm() -
                                 axis * axis.transpose() / axis.norm() / axis.norm() / axis.norm());

        hessian->block<3, 3>(0, 6) += crossMatrix(v) * dahat / v.squaredNorm();
        hessian->block<3, 3>(3, 6) += -crossMatrix(w) * dahat / w.squaredNorm();
    }

    return theta;
}

double cosTriangleAngle(const MeshConnectivity& mesh,
                        const Eigen::MatrixXd& curPos,
                        int face,
                        int vid,
                        Eigen::Matrix<double, 1, 9>* derivative,
                        Eigen::Matrix<double, 9, 9>* hessian) {
    Eigen::Vector3d e0 = curPos.row(mesh.faceVertex(face, (vid + 1) % 3)).transpose() -
                         curPos.row(mesh.faceVertex(face, vid)).transpose();
    Eigen::Vector3d e1 = curPos.row(mesh.faceVertex(face, (vid + 2) % 3)).transpose() -
                         curPos.row(mesh.faceVertex(face, vid)).transpose();

    Eigen::Vector3d de0_norm, de1_norm;
    Eigen::Matrix3d d2e0_norm, d2e1_norm;

    double e0_norm = vec_norm(e0, (derivative || hessian) ? &de0_norm : nullptr,
                                 hessian ? &d2e0_norm : nullptr);
    double e1_norm = vec_norm(e1, (derivative || hessian) ? &de1_norm : nullptr,
                                 hessian ? &d2e1_norm : nullptr);

    double e0_dot_e1 = e0.dot(e1);
    double e0e1_norm = e0_norm * e1_norm;

    double cos = e0_dot_e1 / e0e1_norm;

    if(derivative || hessian) {
        Eigen::Matrix<double, 1, 9> e0_norm_deriv;
        e0_norm_deriv.setZero();
        e0_norm_deriv.block<1, 3>(0, 3 * vid) = -de0_norm.transpose();
        e0_norm_deriv.block<1, 3>(0, 3 * ((vid + 1) % 3)) = de0_norm.transpose();

        Eigen::Matrix<double, 1, 9> e1_norm_deriv;
        e1_norm_deriv.setZero();
        e1_norm_deriv.block<1, 3>(0, 3 * vid) = -de1_norm.transpose();
        e1_norm_deriv.block<1, 3>(0, 3 * ((vid + 2) % 3)) = de1_norm.transpose();

        Eigen::Matrix<double, 1, 9> e0e1_norm_deriv = e0_norm * e1_norm_deriv + e1_norm * e0_norm_deriv;
        Eigen::Matrix<double, 1, 9> e0_dot_e1_deriv;
        e0_dot_e1_deriv.block<1, 3>(0, 3 * vid) = -(e0 + e1).transpose();
        e0_dot_e1_deriv.block<1, 3>(0, 3 * ((vid + 1) % 3)) = e1.transpose();
        e0_dot_e1_deriv.block<1, 3>(0, 3 * ((vid + 2) % 3)) = e0.transpose();

        // cos = e0_dot_e1 / e0e1_norm
        if (derivative) {
            *derivative = (e0e1_norm * e0_dot_e1_deriv - e0_dot_e1 * e0e1_norm_deriv) / (e0e1_norm * e0e1_norm);
        }

        if(hessian) {
            Eigen::Matrix<double, 9, 9> e0_norm_hess;
            e0_norm_hess.setZero();
            e0_norm_hess.block<3, 3>(3 * vid, 3 * vid) = d2e0_norm;
            e0_norm_hess.block<3, 3>(3 * vid, 3 * ((vid + 1) % 3)) = -d2e0_norm;
            e0_norm_hess.block<3, 3>(3 * ((vid + 1) % 3), 3 * vid) = -d2e0_norm;
            e0_norm_hess.block<3, 3>(3 * ((vid + 1) % 3), 3 * ((vid + 1) % 3)) = d2e0_norm;

            Eigen::Matrix<double, 9, 9> e1_norm_hess;
            e1_norm_hess.setZero();
            e1_norm_hess.block<3, 3>(3 * vid, 3 * vid) = d2e1_norm;
            e1_norm_hess.block<3, 3>(3 * vid, 3 * ((vid + 2) % 3)) = -d2e1_norm;
            e1_norm_hess.block<3, 3>(3 * ((vid + 2) % 3), 3 * vid) = -d2e1_norm;
            e1_norm_hess.block<3, 3>(3 * ((vid + 2) % 3), 3 * ((vid + 2) % 3)) = d2e1_norm;

            Eigen::Matrix<double, 9, 9> e0e1_norm_hess = e0_norm * e1_norm_hess + e1_norm * e0_norm_hess;
            e0e1_norm_hess += e0_norm_deriv.transpose() * e1_norm_deriv +
                              e1_norm_deriv.transpose() * e0_norm_deriv;

            Eigen::Matrix<double, 9, 9> e0_dot_e1_hess;
            Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
            e0_dot_e1_hess.setZero();
            e0_dot_e1_hess.block<3, 3>(3 * vid, 3 * vid) = 2 * id;
            e0_dot_e1_hess.block<3, 3>(3 * vid, 3 * ((vid + 1) % 3)) = -id;
            e0_dot_e1_hess.block<3, 3>(3 * vid, 3 * ((vid + 2) % 3)) = -id;

            e0_dot_e1_hess.block<3, 3>(3 * ((vid + 1) % 3), 3 * vid) = -id;
            e0_dot_e1_hess.block<3, 3>(3 * ((vid + 1) % 3), 3 * ((vid + 2) % 3)) = id;

            e0_dot_e1_hess.block<3, 3>(3 * ((vid + 2) % 3), 3 * vid) = -id;
            e0_dot_e1_hess.block<3, 3>(3 * ((vid + 2) % 3), 3 * ((vid + 1) % 3)) = id;

            // cos = e0_dot_e1 / e0e1_norm
            *hessian = -1 / (e0e1_norm * e0e1_norm) * (e0e1_norm_deriv.transpose() * e0_dot_e1_deriv +
                                                  e0_dot_e1_deriv.transpose() * e0e1_norm_deriv + e0_dot_e1 * e0e1_norm_hess);
            (*hessian) += 2 * e0_dot_e1 / (e0e1_norm * e0e1_norm * e0e1_norm) * (e0e1_norm_deriv.transpose() * e0e1_norm_deriv);
            (*hessian) += 1.0 / e0e1_norm * e0_dot_e1_hess;
        }
    }

    // Clamp cos to [-1, 1] to avoid NaN from acos
    cos = std::max(-1.0, std::min(1.0, cos));

    return cos;
}


Eigen::Vector3d faceNormal(const MeshConnectivity& mesh,
                           const Eigen::MatrixXd& curPos,
                           int face,
                           int startidx,
                           Eigen::Matrix<double, 3, 9>* derivative,
                           std::vector<Eigen::Matrix<double, 9, 9>>* hessian) {
    if (derivative) derivative->setZero();

    if (hessian) {
        hessian->resize(3);
        for (int i = 0; i < 3; i++) (*hessian)[i].setZero();
    }

    int v0 = startidx % 3;
    int v1 = (startidx + 1) % 3;
    int v2 = (startidx + 2) % 3;
    Eigen::Vector3d qi0 = curPos.row(mesh.faceVertex(face, v0)).transpose();
    Eigen::Vector3d qi1 = curPos.row(mesh.faceVertex(face, v1)).transpose();
    Eigen::Vector3d qi2 = curPos.row(mesh.faceVertex(face, v2)).transpose();
    Eigen::Vector3d n = (qi1 - qi0).cross(qi2 - qi0);

    if (derivative) {
        derivative->block(0, 0, 3, 3) += crossMatrix(qi2 - qi1);
        derivative->block(0, 3, 3, 3) += crossMatrix(qi0 - qi2);
        derivative->block(0, 6, 3, 3) += crossMatrix(qi1 - qi0);
    }

    if (hessian) {
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3d ej(0, 0, 0);
            ej[j] = 1.0;
            Eigen::Matrix3d ejc = crossMatrix(ej);
            (*hessian)[j].block(0, 3, 3, 3) -= ejc;
            (*hessian)[j].block(0, 6, 3, 3) += ejc;
            (*hessian)[j].block(3, 6, 3, 3) -= ejc;
            (*hessian)[j].block(3, 0, 3, 3) += ejc;
            (*hessian)[j].block(6, 0, 3, 3) -= ejc;
            (*hessian)[j].block(6, 3, 3, 3) += ejc;
        }
    }

    return n;
}


double triangleAltitude(const MeshConnectivity& mesh,
                        const Eigen::MatrixXd& curPos,
                        int face,
                        int edgeidx,
                        Eigen::Matrix<double, 1, 9>* derivative,
                        Eigen::Matrix<double, 9, 9>* hessian) {
    if (derivative) derivative->setZero();
    if (hessian) hessian->setZero();

    Eigen::Matrix<double, 3, 9> nderiv;
    std::vector<Eigen::Matrix<double, 9, 9>> nhess;
    Eigen::Vector3d n =
        faceNormal(mesh, curPos, face, edgeidx, (derivative || hessian ? &nderiv : NULL), hessian ? &nhess : NULL);

    int v2 = (edgeidx + 2) % 3;
    int v1 = (edgeidx + 1) % 3;
    Eigen::Vector3d q2 = curPos.row(mesh.faceVertex(face, v2)).transpose();
    Eigen::Vector3d q1 = curPos.row(mesh.faceVertex(face, v1)).transpose();

    Eigen::Vector3d e = q2 - q1;
    double nnorm = n.norm();
    double enorm = e.norm();
    double h = nnorm / enorm;

    if (derivative) {
        for (int i = 0; i < 3; i++) {
            *derivative += nderiv.row(i) * n[i] / nnorm / enorm;
        }
        derivative->block(0, 6, 1, 3) += -nnorm / enorm / enorm / enorm * e.transpose();
        derivative->block(0, 3, 1, 3) += nnorm / enorm / enorm / enorm * e.transpose();
    }

    if (hessian) {
        for (int i = 0; i < 3; i++) {
            *hessian += nhess[i] * n[i] / nnorm / enorm;
        }
        Eigen::Matrix3d P = Eigen::Matrix3d::Identity() / nnorm - n * n.transpose() / nnorm / nnorm / nnorm;
        *hessian += nderiv.transpose() * P * nderiv / enorm;
        hessian->block(6, 0, 3, 9) += -e * n.transpose() * nderiv / nnorm / enorm / enorm / enorm;
        hessian->block(3, 0, 3, 9) += e * n.transpose() * nderiv / nnorm / enorm / enorm / enorm;
        hessian->block(0, 6, 9, 3) += -nderiv.transpose() * n * e.transpose() / nnorm / enorm / enorm / enorm;
        hessian->block(0, 3, 9, 3) += nderiv.transpose() * n * e.transpose() / nnorm / enorm / enorm / enorm;
        hessian->block(6, 6, 3, 3) += -nnorm / enorm / enorm / enorm * Eigen::Matrix3d::Identity();
        hessian->block(6, 3, 3, 3) += nnorm / enorm / enorm / enorm * Eigen::Matrix3d::Identity();
        hessian->block(3, 6, 3, 3) += nnorm / enorm / enorm / enorm * Eigen::Matrix3d::Identity();
        hessian->block(3, 3, 3, 3) += -nnorm / enorm / enorm / enorm * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d outer = e * e.transpose() * 3.0 * nnorm / enorm / enorm / enorm / enorm / enorm;
        hessian->block(6, 6, 3, 3) += outer;
        hessian->block(6, 3, 3, 3) += -outer;
        hessian->block(3, 6, 3, 3) += -outer;
        hessian->block(3, 3, 3, 3) += outer;
    }

    return h;
}

Eigen::Matrix2d firstFundamentalForm(const MeshConnectivity& mesh,
                                     const Eigen::MatrixXd& curPos,
                                     int face,
                                     Eigen::Matrix<double, 4, 9>* derivative,  // F(face, i)
                                     std::vector<Eigen::Matrix<double, 9, 9>>* hessian) {
    Eigen::Vector3d q0 = curPos.row(mesh.faceVertex(face, 0));
    Eigen::Vector3d q1 = curPos.row(mesh.faceVertex(face, 1));
    Eigen::Vector3d q2 = curPos.row(mesh.faceVertex(face, 2));
    Eigen::Matrix2d result;
    result << (q1 - q0).dot(q1 - q0), (q1 - q0).dot(q2 - q0), (q2 - q0).dot(q1 - q0), (q2 - q0).dot(q2 - q0);

    if (derivative) {
        derivative->setZero();
        derivative->block<1, 3>(0, 3) += 2.0 * (q1 - q0).transpose();
        derivative->block<1, 3>(0, 0) -= 2.0 * (q1 - q0).transpose();
        derivative->block<1, 3>(1, 6) += (q1 - q0).transpose();
        derivative->block<1, 3>(1, 3) += (q2 - q0).transpose();
        derivative->block<1, 3>(1, 0) += -(q1 - q0).transpose() - (q2 - q0).transpose();
        derivative->block<1, 3>(2, 6) += (q1 - q0).transpose();
        derivative->block<1, 3>(2, 3) += (q2 - q0).transpose();
        derivative->block<1, 3>(2, 0) += -(q1 - q0).transpose() - (q2 - q0).transpose();
        derivative->block<1, 3>(3, 6) += 2.0 * (q2 - q0).transpose();
        derivative->block<1, 3>(3, 0) -= 2.0 * (q2 - q0).transpose();
    }

    if (hessian) {
        hessian->resize(4);
        for (int i = 0; i < 4; i++) {
            (*hessian)[i].setZero();
        }
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        (*hessian)[0].block<3, 3>(0, 0) += 2.0 * I;
        (*hessian)[0].block<3, 3>(3, 3) += 2.0 * I;
        (*hessian)[0].block<3, 3>(0, 3) -= 2.0 * I;
        (*hessian)[0].block<3, 3>(3, 0) -= 2.0 * I;

        (*hessian)[1].block<3, 3>(3, 6) += I;
        (*hessian)[1].block<3, 3>(6, 3) += I;
        (*hessian)[1].block<3, 3>(0, 3) -= I;
        (*hessian)[1].block<3, 3>(0, 6) -= I;
        (*hessian)[1].block<3, 3>(3, 0) -= I;
        (*hessian)[1].block<3, 3>(6, 0) -= I;
        (*hessian)[1].block<3, 3>(0, 0) += 2.0 * I;

        (*hessian)[2].block<3, 3>(3, 6) += I;
        (*hessian)[2].block<3, 3>(6, 3) += I;
        (*hessian)[2].block<3, 3>(0, 3) -= I;
        (*hessian)[2].block<3, 3>(0, 6) -= I;
        (*hessian)[2].block<3, 3>(3, 0) -= I;
        (*hessian)[2].block<3, 3>(6, 0) -= I;
        (*hessian)[2].block<3, 3>(0, 0) += 2.0 * I;

        (*hessian)[3].block<3, 3>(0, 0) += 2.0 * I;
        (*hessian)[3].block<3, 3>(6, 6) += 2.0 * I;
        (*hessian)[3].block<3, 3>(0, 6) -= 2.0 * I;
        (*hessian)[3].block<3, 3>(6, 0) -= 2.0 * I;
    }

    return result;
}

double edgeTheta(const MeshConnectivity& mesh,
                 const Eigen::MatrixXd& curPos,
                 int edge,
                 Eigen::Matrix<double, 1, 12>* derivative,  // edgeVertex, then edgeOppositeVertex
                 Eigen::Matrix<double, 12, 12>* hessian) {
    if (derivative) derivative->setZero();
    if (hessian) hessian->setZero();
    int v0 = mesh.edgeVertex(edge, 0);
    int v1 = mesh.edgeVertex(edge, 1);
    int v2 = mesh.edgeOppositeVertex(edge, 0);
    int v3 = mesh.edgeOppositeVertex(edge, 1);
    if (v2 == -1 || v3 == -1) return 0;  // boundary edge

    Eigen::Vector3d q0 = curPos.row(v0);
    Eigen::Vector3d q1 = curPos.row(v1);
    Eigen::Vector3d q2 = curPos.row(v2);
    Eigen::Vector3d q3 = curPos.row(v3);

    Eigen::Vector3d n0 = (q0 - q2).cross(q1 - q2);
    Eigen::Vector3d n1 = (q1 - q3).cross(q0 - q3);
    Eigen::Vector3d axis = q1 - q0;
    Eigen::Matrix<double, 1, 9> angderiv;
    Eigen::Matrix<double, 9, 9> anghess;

    double theta = angle(n0, n1, axis, (derivative || hessian) ? &angderiv : NULL, hessian ? &anghess : NULL);

    if (derivative) {
        derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 0) * crossMatrix(q2 - q1);
        derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 0) * crossMatrix(q0 - q2);
        derivative->block<1, 3>(0, 6) += angderiv.block<1, 3>(0, 0) * crossMatrix(q1 - q0);

        derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 3) * crossMatrix(q1 - q3);
        derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 3) * crossMatrix(q3 - q0);
        derivative->block<1, 3>(0, 9) += angderiv.block<1, 3>(0, 3) * crossMatrix(q0 - q1);
    }

    if (hessian) {
        Eigen::Matrix3d vqm[3];
        vqm[0] = crossMatrix(q0 - q2);
        vqm[1] = crossMatrix(q1 - q0);
        vqm[2] = crossMatrix(q2 - q1);
        Eigen::Matrix3d wqm[3];
        wqm[0] = crossMatrix(q0 - q1);
        wqm[1] = crossMatrix(q1 - q3);
        wqm[2] = crossMatrix(q3 - q0);

        int vindices[3] = {3, 6, 0};
        int windices[3] = {9, 0, 3};

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                hessian->block<3, 3>(vindices[i], vindices[j]) +=
                    vqm[i].transpose() * anghess.block<3, 3>(0, 0) * vqm[j];
                hessian->block<3, 3>(vindices[i], windices[j]) +=
                    vqm[i].transpose() * anghess.block<3, 3>(0, 3) * wqm[j];
                hessian->block<3, 3>(windices[i], vindices[j]) +=
                    wqm[i].transpose() * anghess.block<3, 3>(3, 0) * vqm[j];
                hessian->block<3, 3>(windices[i], windices[j]) +=
                    wqm[i].transpose() * anghess.block<3, 3>(3, 3) * wqm[j];
            }

            hessian->block<3, 3>(vindices[i], 3) += vqm[i].transpose() * anghess.block<3, 3>(0, 6);
            hessian->block<3, 3>(3, vindices[i]) += anghess.block<3, 3>(6, 0) * vqm[i];
            hessian->block<3, 3>(vindices[i], 0) += -vqm[i].transpose() * anghess.block<3, 3>(0, 6);
            hessian->block<3, 3>(0, vindices[i]) += -anghess.block<3, 3>(6, 0) * vqm[i];

            hessian->block<3, 3>(windices[i], 3) += wqm[i].transpose() * anghess.block<3, 3>(3, 6);
            hessian->block<3, 3>(3, windices[i]) += anghess.block<3, 3>(6, 3) * wqm[i];
            hessian->block<3, 3>(windices[i], 0) += -wqm[i].transpose() * anghess.block<3, 3>(3, 6);
            hessian->block<3, 3>(0, windices[i]) += -anghess.block<3, 3>(6, 3) * wqm[i];
        }

        Eigen::Vector3d dang1 = angderiv.block<1, 3>(0, 0).transpose();
        Eigen::Vector3d dang2 = angderiv.block<1, 3>(0, 3).transpose();

        Eigen::Matrix3d dang1mat = crossMatrix(dang1);
        Eigen::Matrix3d dang2mat = crossMatrix(dang2);

        hessian->block<3, 3>(6, 3) += dang1mat;
        hessian->block<3, 3>(0, 3) -= dang1mat;
        hessian->block<3, 3>(0, 6) += dang1mat;
        hessian->block<3, 3>(3, 0) += dang1mat;
        hessian->block<3, 3>(3, 6) -= dang1mat;
        hessian->block<3, 3>(6, 0) -= dang1mat;

        hessian->block<3, 3>(9, 0) += dang2mat;
        hessian->block<3, 3>(3, 0) -= dang2mat;
        hessian->block<3, 3>(3, 9) += dang2mat;
        hessian->block<3, 3>(0, 3) += dang2mat;
        hessian->block<3, 3>(0, 9) -= dang2mat;
        hessian->block<3, 3>(9, 3) -= dang2mat;
    }

    return theta;
}

double vec_norm(const Eigen::Vector3d& v, Eigen::Vector3d* derivative, Eigen::Matrix3d* hessian) {
    double vnorm = v.norm();

    if(vnorm == 0) {
        if(derivative) {
            derivative->setZero();
        }
        if(hessian) {
            hessian->setZero();
        }
        return 0;
    }

    if (derivative) {
        *derivative = v.normalized();
    }
    if (hessian) {
        Eigen::Matrix3d id;
        id.setIdentity();
        *hessian = 1.0 / vnorm * id - v * v.transpose() / (vnorm * vnorm * vnorm);
    }

    return vnorm;
}

Eigen::Vector3d normalizedVector(const Eigen::Vector3d& v,
                                 Eigen::Matrix3d* derivative,
                                 std::vector<Eigen::Matrix3d>* hessian) {
    if (derivative) {
        derivative->setZero();
    }
    if(hessian) {
        hessian->resize(3, Eigen::Matrix3d::Zero());
    }

    Eigen::Vector3d norm_deriv;
    Eigen::Matrix3d norm_hess;
    double vnorm = vec_norm(v, derivative || hessian ? &norm_deriv : nullptr,
                            hessian ? &norm_hess : nullptr);
    if(!vnorm) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d vhat = v / vnorm;
    if (derivative) {
        for(int i = 0; i < 3; i++) {
            derivative->row(i) = -v[i] / vnorm / vnorm * norm_deriv.transpose();
            (*derivative)(i, i) += 1.0 / vnorm;
        }
    }

    if (hessian) {
        for(int i = 0; i < 3; i++) {
            Eigen::Vector3d ei(0, 0, 0);
            ei[i] = 1.0;
            (*hessian)[i] = 2 * v[i] / vnorm / vnorm / vnorm * norm_deriv * norm_deriv.transpose() - v[i] / vnorm / vnorm * norm_hess;
            (*hessian)[i] += -1 / vnorm / vnorm * (ei * norm_deriv.transpose() + norm_deriv * ei.transpose());
        }
    }

    return vhat;
}


};  // namespace LibShell