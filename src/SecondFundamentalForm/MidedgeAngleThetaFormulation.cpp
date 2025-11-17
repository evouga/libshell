#include "../../include/MidedgeAngleThetaFormulation.h"
#include <Eigen/Geometry>
#include "../GeometryDerivatives.h"
#include "../../include/MeshConnectivity.h"
#include <iostream>
#include <random>
#include <Eigen/Geometry>

namespace LibShell {

    static double edgeTheta(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        int edge,
        Eigen::Matrix<double, 1, 12>* derivative, // edgeVertex, then edgeOppositeVertex
        Eigen::Matrix<double, 12, 12>* hessian)
    {
        if (derivative)
            derivative->setZero();
        if (hessian)
            hessian->setZero();
        int v0 = mesh.edgeVertex(edge, 0);
        int v1 = mesh.edgeVertex(edge, 1);
        int v2 = mesh.edgeOppositeVertex(edge, 0);
        int v3 = mesh.edgeOppositeVertex(edge, 1);
        if (v2 == -1 || v3 == -1)
            return 0; // boundary edge

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

        if (derivative)
        {
            derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 0) * crossMatrix(q2 - q1);
            derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 0) * crossMatrix(q0 - q2);
            derivative->block<1, 3>(0, 6) += angderiv.block<1, 3>(0, 0) * crossMatrix(q1 - q0);

            derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 3) * crossMatrix(q1 - q3);
            derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 3) * crossMatrix(q3 - q0);
            derivative->block<1, 3>(0, 9) += angderiv.block<1, 3>(0, 3) * crossMatrix(q0 - q1);
        }

        if (hessian)
        {
            Eigen::Matrix3d vqm[3];
            vqm[0] = crossMatrix(q0 - q2);
            vqm[1] = crossMatrix(q1 - q0);
            vqm[2] = crossMatrix(q2 - q1);
            Eigen::Matrix3d wqm[3];
            wqm[0] = crossMatrix(q0 - q1);
            wqm[1] = crossMatrix(q1 - q3);
            wqm[2] = crossMatrix(q3 - q0);

            int vindices[3] = { 3, 6, 0 };
            int windices[3] = { 9, 0, 3 };

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    hessian->block<3, 3>(vindices[i], vindices[j]) += vqm[i].transpose() * anghess.block<3, 3>(0, 0) * vqm[j];
                    hessian->block<3, 3>(vindices[i], windices[j]) += vqm[i].transpose() * anghess.block<3, 3>(0, 3) * wqm[j];
                    hessian->block<3, 3>(windices[i], vindices[j]) += wqm[i].transpose() * anghess.block<3, 3>(3, 0) * vqm[j];
                    hessian->block<3, 3>(windices[i], windices[j]) += wqm[i].transpose() * anghess.block<3, 3>(3, 3) * wqm[j];
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

    static Eigen::Vector3d secondFundamentalFormEntries(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& edgeThetas,
        int face,
        Eigen::Matrix<double, 3, 21>* derivative,
        std::vector<Eigen::Matrix<double, 21, 21> >* hessian)
    {
        if (derivative)
            derivative->setZero();
        if (hessian)
        {
            hessian->resize(3);
            for (int i = 0; i < 3; i++)
                (*hessian)[i].setZero();
        }

        Eigen::Vector3d II;
        for (int i = 0; i < 3; i++)
        {
            Eigen::Matrix<double, 1, 9> hderiv;
            Eigen::Matrix<double, 9, 9> hhess;
            double altitude = triangleAltitude(mesh, curPos, face, i, (derivative || hessian) ? &hderiv : NULL, hessian ? &hhess : NULL);

            int edge = mesh.faceEdge(face, i);
            Eigen::Matrix<double, 1, 12> thetaderiv;
            Eigen::Matrix<double, 12, 12> thetahess;
            double theta = edgeTheta(mesh, curPos, edge, (derivative || hessian) ? &thetaderiv : NULL, hessian ? &thetahess : NULL);

            double orient = mesh.faceEdgeOrientation(face, i) == 0 ? 1.0 : -1.0;
            double alpha = 0.5 * theta + orient * edgeThetas[edge];
            II[i] = 2.0 * altitude * alpha;

            if (derivative)
            {
                int hv0 = i;
                int hv1 = (i + 1) % 3;
                int hv2 = (i + 2) % 3;
                derivative->block(i, 3 * hv0, 1, 3) += 2.0 * alpha * hderiv.block(0, 0, 1, 3);
                derivative->block(i, 3 * hv1, 1, 3) += 2.0 * alpha * hderiv.block(0, 3, 1, 3);
                derivative->block(i, 3 * hv2, 1, 3) += 2.0 * alpha * hderiv.block(0, 6, 1, 3);

                int av0, av1, av2, av3;
                if (mesh.faceEdgeOrientation(face, i) == 0)
                {
                    av0 = (i + 1) % 3;
                    av1 = (i + 2) % 3;
                    av2 = i;
                    av3 = 3 + i;
                }
                else
                {
                    av0 = (i + 2) % 3;
                    av1 = (i + 1) % 3;
                    av2 = 3 + i;
                    av3 = i;
                }
                derivative->block(i, 3 * av0, 1, 3) += altitude * thetaderiv.block(0, 0, 1, 3);
                derivative->block(i, 3 * av1, 1, 3) += altitude * thetaderiv.block(0, 3, 1, 3);
                derivative->block(i, 3 * av2, 1, 3) += altitude * thetaderiv.block(0, 6, 1, 3);
                derivative->block(i, 3 * av3, 1, 3) += altitude * thetaderiv.block(0, 9, 1, 3);
                (*derivative)(i, 18 + i) += 2.0 * altitude * orient;

            }

            if (hessian)
            {
                int hv[3];
                hv[0] = i;
                hv[1] = (i + 1) % 3;
                hv[2] = (i + 2) % 3;
                for (int j = 0; j < 3; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        (*hessian)[i].block(3 * hv[j], 3 * hv[k], 3, 3) += 2.0 * alpha * hhess.block(3 * j, 3 * k, 3, 3);
                    }
                }

                int av[4];
                if (mesh.faceEdgeOrientation(face, i) == 0)
                {
                    av[0] = (i + 1) % 3;
                    av[1] = (i + 2) % 3;
                    av[2] = i;
                    av[3] = 3 + i;
                }
                else
                {
                    av[0] = (i + 2) % 3;
                    av[1] = (i + 1) % 3;
                    av[2] = 3 + i;
                    av[3] = i;
                }

                for (int k = 0; k < 3; k++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        (*hessian)[i].block(3 * av[j], 3 * hv[k], 3, 3) += thetaderiv.block(0, 3 * j, 1, 3).transpose() * hderiv.block(0, 3 * k, 1, 3);
                        (*hessian)[i].block(3 * hv[k], 3 * av[j], 3, 3) += hderiv.block(0, 3 * k, 1, 3).transpose() * thetaderiv.block(0, 3 * j, 1, 3);
                    }
                    (*hessian)[i].block(18 + i, 3 * hv[k], 1, 3) += 2.0 * orient * hderiv.block(0, 3 * k, 1, 3);
                    (*hessian)[i].block(3 * hv[k], 18 + i, 3, 1) += 2.0 * orient * hderiv.block(0, 3 * k, 1, 3).transpose();
                }

                for (int k = 0; k < 4; k++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        (*hessian)[i].block(3 * av[j], 3 * av[k], 3, 3) += altitude * thetahess.block(3 * j, 3 * k, 3, 3);                        
                    }
                }
            }
        }

        return II;
    }


    Eigen::Matrix2d MidedgeAngleThetaFormulation::secondFundamentalForm(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        int face,
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs>* derivative,
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs> >* hessian)
    {
        if (derivative)
        {
            derivative->resize(4, 21);
            derivative->setZero();
        }
        if (hessian)
        {
            hessian->resize(4);
            for (int i = 0; i < 4; i++)
            {
                (*hessian)[i].resize(21, 21);
                (*hessian)[i].setZero();
            }
        }


        Eigen::Matrix<double, 3, 21> IIderiv;
        std::vector < Eigen::Matrix<double, 21, 21> > IIhess;

        Eigen::Vector3d II = secondFundamentalFormEntries(mesh, curPos, extraDOFs, face, derivative ? &IIderiv : NULL, hessian ? &IIhess : NULL);

        Eigen::Matrix2d result;
        result << II[0] + II[1], II[0], II[0], II[0] + II[2];

        if (derivative)
        {
            derivative->row(0) += IIderiv.row(0);
            derivative->row(0) += IIderiv.row(1);

            derivative->row(1) += IIderiv.row(0);
            derivative->row(2) += IIderiv.row(0);

            derivative->row(3) += IIderiv.row(0);
            derivative->row(3) += IIderiv.row(2);
        }
        if (hessian)
        {
            (*hessian)[0] += IIhess[0];
            (*hessian)[0] += IIhess[1];

            (*hessian)[1] += IIhess[0];
            (*hessian)[2] += IIhess[0];

            (*hessian)[3] += IIhess[0];
            (*hessian)[3] += IIhess[2];
        }

        return result;
    }

    constexpr int MidedgeAngleThetaFormulation::numExtraDOFs;

    void MidedgeAngleThetaFormulation::initializeExtraDOFs(Eigen::VectorXd& extraDOFs, const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos)
    {
        extraDOFs.resize(mesh.nEdges());
        extraDOFs.setZero();
    }

};