#include "../../include/MidedgeAverageFormulation.h"
#include "../GeometryDerivatives.h"
#include "../../include/MeshConnectivity.h"
#include <iostream>
#include <random>

static Eigen::Vector3d secondFundamentalFormEntries(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    int face,
    Eigen::Matrix<double, 3, 18> *derivative,
    std::vector<Eigen::Matrix<double, 18, 18> > *hessian)
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

    Eigen::Vector3d oppNormals[3];
    Eigen::Matrix<double, 3, 9> dn[3];
    std::vector<Eigen::Matrix<double, 9, 9> > hn[3];

    Eigen::Matrix<double, 3, 9> dcn;
    std::vector<Eigen::Matrix<double, 9, 9> > hcn;
    Eigen::Vector3d cNormal = faceNormal(mesh, curPos, face, 0, (derivative || hessian) ? &dcn : NULL, hessian ? &hcn : NULL);
    
    for (int i = 0; i < 3; i++)
    {
        int oppidx = mesh.vertexOppositeFaceEdge(face, i);
        int edge = mesh.faceEdge(face, i);
        int oppface = mesh.edgeFace(edge, 1 - mesh.faceEdgeOrientation(face, i));
        if (oppface == -1)
        {
            oppNormals[i].setZero();
            dn[i].setZero();
            hn[i].resize(3);
            for (int j = 0; j < 3; j++)
                hn[i][j].setZero();
        }
        else
        {
            int idx = 0;
            for (int j = 0; j < 3; j++)
            {
                if (mesh.faceVertex(oppface, j) == oppidx)
                    idx = j;
            }
            oppNormals[i] = faceNormal(mesh, curPos, oppface, idx, (derivative || hessian) ? &dn[i] : NULL, hessian ? &hn[i] : NULL);
        }
    }

    Eigen::Vector3d qs[3];
    double mnorms[3];
    for (int i = 0; i < 3; i++)
    {
        qs[i] = curPos.row(mesh.faceVertex(face, i)).transpose();
        mnorms[i] = (oppNormals[i] + cNormal).norm();
    }

    for (int i = 0; i < 3; i++)
    {
        int ip1 = (i + 1) % 3;
        int ip2 = (i + 2) % 3;
        II[i] = (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i];
        if (derivative)
        {
            derivative->block(i, 3*i, 1, 3) += -2.0 * oppNormals[i].transpose() / mnorms[i];
            derivative->block(i, 3*ip1, 1, 3) += 1.0 * oppNormals[i].transpose() / mnorms[i];
            derivative->block(i, 3*ip2, 1, 3) += 1.0 * oppNormals[i].transpose() / mnorms[i];
            
            derivative->block(i, 9 + 3*i, 1, 3) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].block(0, 0, 3, 3);
            derivative->block(i, 3*ip2, 1, 3) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].block(0, 3, 3, 3);
            derivative->block(i, 3*ip1, 1, 3) += (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() / mnorms[i] * dn[i].block(0, 6, 3, 3);

            derivative->block(i, 9 + 3*i, 1, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 0, 3, 3);
            derivative->block(i, 3*ip2, 1, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3, 3, 3);
            derivative->block(i, 3*ip1, 1, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 6, 3, 3);

            derivative->block(i, 0, 1, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 0, 3, 3);
            derivative->block(i, 3, 1, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3, 3, 3);
            derivative->block(i, 6, 1, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 6, 3, 3);
        }
        if (hessian)
        {
            int ip1 = (i + 1) % 3;
            int ip2 = (i + 2) % 3;

            int miidx[3];
            miidx[0] = 9 + 3 * i;
            miidx[1] = 3 * ((i + 2) % 3);
            miidx[2] = 3 * ((i + 1) % 3);

            for (int j = 0; j < 3; j++)
            {
                (*hessian)[i].block(miidx[j], 3 * ip1, 3, 3) += dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i];
                (*hessian)[i].block(miidx[j], 3 * ip2, 3, 3) += dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i];
                (*hessian)[i].block(miidx[j], 3 * i, 3, 3) += -2.0 * dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i];

                (*hessian)[i].block(miidx[j], 3 * ip1, 3, 3) += -dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                (*hessian)[i].block(miidx[j], 3 * ip2, 3, 3) += -dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                (*hessian)[i].block(miidx[j], 3 * i, 3, 3) += 2.0 * dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();

                (*hessian)[i].block(3 * j, 3 * ip1, 3, 3) += -dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                (*hessian)[i].block(3 * j, 3 * ip2, 3, 3) += -dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();
                (*hessian)[i].block(3 * j, 3 * i, 3, 3) += 2.0 * dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal)*oppNormals[i].transpose();

                (*hessian)[i].block(3 * ip1, miidx[j], 3, 3) += dn[i].block(0, 3 * j, 3, 3) / mnorms[i];
                (*hessian)[i].block(3 * ip2, miidx[j], 3, 3) += dn[i].block(0, 3 * j, 3, 3) / mnorms[i];
                (*hessian)[i].block(3 * i, miidx[j], 3, 3) += -2.0 * dn[i].block(0, 3 * j, 3, 3) / mnorms[i];

                for (int k = 0; k < 3; k++)
                {
                    (*hessian)[i].block(miidx[j], miidx[k], 3, 3) += -dn[i].block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal) * (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() * dn[i].block(0, 3 * k, 3, 3);                    
                    (*hessian)[i].block(3 * j, miidx[k], 3, 3) += -dcn.block(0, 3 * j, 3, 3).transpose() / mnorms[i] / mnorms[i] / mnorms[i] *  (oppNormals[i] + cNormal) * (qs[ip1] + qs[ip2] - 2.0*qs[i]).transpose() * dn[i].block(0, 3 * k, 3, 3);
                }

                (*hessian)[i].block(3 * ip1, miidx[j], 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * j, 3, 3);
                (*hessian)[i].block(3 * ip2, miidx[j], 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * j, 3, 3);
                (*hessian)[i].block(3 * i, miidx[j], 3, 3) += 2.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * j, 3, 3);

                (*hessian)[i].block(3 * ip1, 3 * j, 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * j, 3, 3);
                (*hessian)[i].block(3 * ip2, 3 * j, 3, 3) += -1.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * j, 3, 3);
                (*hessian)[i].block(3 * i, 3 * j, 3, 3) += 2.0 / mnorms[i] / mnorms[i] / mnorms[i] *  oppNormals[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * j, 3, 3);

                for (int k = 0; k < 3; k++)
                {
                    (*hessian)[i].block(miidx[j], miidx[k], 3, 3) += -dn[i].block(0, 3*j, 3, 3).transpose() * (qs[ip1] + qs[ip2] - 2.0*qs[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(miidx[j], 3*k, 3, 3) += -dn[i].block(0, 3*j, 3, 3).transpose() * (qs[ip1] + qs[ip2] - 2.0*qs[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * k, 3, 3);
                }

                for (int k = 0; k < 3; k++)
                {
                    (*hessian)[i].block(miidx[j], miidx[k], 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * dn[i].block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(miidx[j], 3*k, 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * dcn.block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(3*j, miidx[k], 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * dn[i].block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(3*j, 3*k, 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * dcn.block(0, 3 * k, 3, 3);

                    (*hessian)[i].block(miidx[j], miidx[k], 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(miidx[j], 3 * k, 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dn[i].block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(3 * j, miidx[k], 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dn[i].block(0, 3 * k, 3, 3);
                    (*hessian)[i].block(3 * j, 3 * k, 3, 3) += 3.0 * (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] * dcn.block(0, 3 * j, 3, 3).transpose() * (oppNormals[i] + cNormal) * (oppNormals[i] + cNormal).transpose() * dcn.block(0, 3 * k, 3, 3);

                    for (int l = 0; l < 3; l++)
                    {
                        (*hessian)[i].block(miidx[j], miidx[k], 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal)[l] * hn[i][l].block(3 * j, 3 * k, 3, 3);
                        (*hessian)[i].block(3*j, 3*k, 3, 3) += -(qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * (oppNormals[i] + cNormal)[l] * hcn[l].block(3 * j, 3 * k, 3, 3);
                        (*hessian)[i].block(miidx[j], miidx[k], 3, 3) += 1.0 / mnorms[i] * (qs[ip1] + qs[ip2] - 2.0*qs[i])[l] * hn[i][l].block(3 * j, 3 * k, 3, 3);
                    }
                }
            }
        }
    }

    return II;
}


Eigen::Matrix2d MidedgeAverageFormulation::secondFundamentalForm(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &extraDOFs,
    int face,
    Eigen::MatrixXd *derivative, 
    std::vector<Eigen::MatrixXd > *hessian) const
{
    if (derivative)
    {
        derivative->resize(4, 18);
        derivative->setZero();
    }
    if (hessian)
    {
        hessian->resize(4);
        for (int i = 0; i < 4; i++)
        {
            (*hessian)[i].resize(18, 18);
            (*hessian)[i].setZero();
        }
    }


    Eigen::Matrix<double, 3, 18> IIderiv;
    std::vector < Eigen::Matrix<double, 18, 18> > IIhess;

    Eigen::Vector3d II = secondFundamentalFormEntries(mesh, curPos, face, derivative ? &IIderiv : NULL, hessian ? &IIhess : NULL);

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

int MidedgeAverageFormulation::numExtraDOFs() const
{
    return 0;
}

void MidedgeAverageFormulation::initializeExtraDOFs(Eigen::VectorXd &extraDOFs, const MeshConnectivity &mesh, const Eigen::MatrixXd &curPos) const
{
    extraDOFs.resize(0);
}