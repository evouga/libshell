#include "../../include/MidedgeAngleSinFormulation.h"
#include <Eigen/Geometry>
#include "../GeometryDerivatives.h"
#include "../../include/MeshConnectivity.h"
#include <iostream>
#include <random>
#include <Eigen/Geometry>

namespace LibShell {

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
            II[i] = 2.0 * altitude * sin(alpha);

            if (derivative)
            {
                int hv0 = i;
                int hv1 = (i + 1) % 3;
                int hv2 = (i + 2) % 3;
                derivative->block(i, 3 * hv0, 1, 3) += 2.0 * sin(alpha) * hderiv.block(0, 0, 1, 3);
                derivative->block(i, 3 * hv1, 1, 3) += 2.0 * sin(alpha) * hderiv.block(0, 3, 1, 3);
                derivative->block(i, 3 * hv2, 1, 3) += 2.0 * sin(alpha) * hderiv.block(0, 6, 1, 3);

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
                derivative->block(i, 3 * av0, 1, 3) += altitude * cos(alpha) * thetaderiv.block(0, 0, 1, 3);
                derivative->block(i, 3 * av1, 1, 3) += altitude * cos(alpha) * thetaderiv.block(0, 3, 1, 3);
                derivative->block(i, 3 * av2, 1, 3) += altitude * cos(alpha) * thetaderiv.block(0, 6, 1, 3);
                derivative->block(i, 3 * av3, 1, 3) += altitude * cos(alpha) * thetaderiv.block(0, 9, 1, 3);
                (*derivative)(i, 18 + i) += 2.0 * altitude * cos(alpha) * orient;

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
                        (*hessian)[i].block(3 * hv[j], 3 * hv[k], 3, 3) += 2.0 * sin(alpha) * hhess.block(3 * j, 3 * k, 3, 3);
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
                        (*hessian)[i].block(3 * av[j], 3 * hv[k], 3, 3) += cos(alpha) * thetaderiv.block(0, 3 * j, 1, 3).transpose() * hderiv.block(0, 3 * k, 1, 3);
                        (*hessian)[i].block(3 * hv[k], 3 * av[j], 3, 3) += cos(alpha) * hderiv.block(0, 3 * k, 1, 3).transpose() * thetaderiv.block(0, 3 * j, 1, 3);
                    }
                    (*hessian)[i].block(18 + i, 3 * hv[k], 1, 3) += 2.0 * cos(alpha) * orient * hderiv.block(0, 3 * k, 1, 3);
                    (*hessian)[i].block(3 * hv[k], 18 + i, 3, 1) += 2.0 * cos(alpha) * orient * hderiv.block(0, 3 * k, 1, 3).transpose();
                }

                for (int k = 0; k < 4; k++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        (*hessian)[i].block(3 * av[j], 3 * av[k], 3, 3) += altitude * cos(alpha) * thetahess.block(3 * j, 3 * k, 3, 3);
                        (*hessian)[i].block(3 * av[j], 3 * av[k], 3, 3) += -0.5 * altitude * sin(alpha) * thetaderiv.block(0, 3 * j, 1, 3).transpose() * thetaderiv.block(0, 3 * k, 1, 3);
                    }
                    (*hessian)[i].block(18 + i, 3 * av[k], 1, 3) += -1.0 * altitude * sin(alpha) * orient * thetaderiv.block(0, 3 * k, 1, 3);
                    (*hessian)[i].block(3 * av[k], 18 + i, 3, 1) += -1.0 * altitude * sin(alpha) * orient * thetaderiv.block(0, 3 * k, 1, 3).transpose();
                }

                (*hessian)[i](18 + i, 18 + i) += -2.0 * altitude * sin(alpha);
            }
        }

        return II;
    }


    Eigen::Matrix2d MidedgeAngleSinFormulation::secondFundamentalForm(
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

    constexpr int MidedgeAngleSinFormulation::numExtraDOFs;

    void MidedgeAngleSinFormulation::initializeExtraDOFs(Eigen::VectorXd& extraDOFs, const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos)
    {
        extraDOFs.resize(mesh.nEdges());
        extraDOFs.setZero();
    }

};