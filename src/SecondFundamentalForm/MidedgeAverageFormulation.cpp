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
    Eigen::Vector3d mvec[3];
    Eigen::Vector3d qvec[3];
    double mnorms[3];
    for (int i = 0; i < 3; i++)
    {
        qs[i] = curPos.row(mesh.faceVertex(face, i)).transpose();
        mvec[i] = oppNormals[i] + cNormal;
        mnorms[i] = mvec[i].norm();
    }
    for (int i = 0; i < 3; i++)
    {
        int ip1 = (i + 1) % 3;
        int ip2 = (i + 2) % 3;
        qvec[i] = qs[ip1] + qs[ip2] - 2.0 * qs[i];
    }

    for (int i = 0; i < 3; i++)
    {
        int ip1 = (i + 1) % 3;
        int ip2 = (i + 2) % 3;
        II[i] = (qs[ip1] + qs[ip2] - 2.0*qs[i]).dot(oppNormals[i]) / mnorms[i];
        if (derivative)
        {
            derivative->block<1,3>(i, 3*i) += -2.0 * oppNormals[i].transpose() / mnorms[i];
            derivative->block<1,3>(i, 3*ip1) += 1.0 * oppNormals[i].transpose() / mnorms[i];
            derivative->block<1,3>(i, 3*ip2) += 1.0 * oppNormals[i].transpose() / mnorms[i];
            
            derivative->block<1,3>(i, 9 + 3*i) += qvec[i].transpose() / mnorms[i] * dn[i].block<3,3>(0, 0);
            derivative->block<1,3>(i, 3*ip2) += qvec[i].transpose() / mnorms[i] * dn[i].block<3,3>(0, 3);
            derivative->block<1,3>(i, 3*ip1) += qvec[i].transpose() / mnorms[i] * dn[i].block<3,3>(0, 6);

            derivative->block<1,3>(i, 9 + 3*i) += -qvec[i].dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * mvec[i].transpose() * dn[i].block<3,3>(0, 0);
            derivative->block<1,3>(i, 3*ip2) += -qvec[i].dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * mvec[i].transpose() * dn[i].block<3,3>(0, 3);
            derivative->block<1,3>(i, 3*ip1) += -qvec[i].dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * mvec[i].transpose() * dn[i].block<3,3>(0, 6);

            derivative->block<1,3>(i, 0) += -qvec[i].dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * mvec[i].transpose() * dcn.block<3,3>(0, 0);
            derivative->block<1,3>(i, 3) += -qvec[i].dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * mvec[i].transpose() * dcn.block<3,3>(0, 3);
            derivative->block<1,3>(i, 6) += -qvec[i].dot(oppNormals[i]) / mnorms[i] / mnorms[i] / mnorms[i] * mvec[i].transpose() * dcn.block<3,3>(0, 6);
        }
        if (hessian)
        {
            int ip1 = (i + 1) % 3;
            int ip2 = (i + 2) % 3;

            int miidx[3];
            miidx[0] = 9 + 3 * i;
            miidx[1] = 3 * ip2;
            miidx[2] = 3 * ip1;

            Eigen::Matrix3d dnij[3];
            for (int j = 0; j < 3; j++)
                dnij[j] = dn[i].block<3, 3>(0, 3 * j);

            for (int j = 0; j < 3; j++)
            {                

                (*hessian)[i].block<3, 3>(miidx[j], 3 * ip1) += (1.0 / mnorms[i]) * dnij[j].transpose();
                (*hessian)[i].block<3, 3>(miidx[j], 3 * ip2) += (1.0 / mnorms[i]) * dnij[j].transpose();
                (*hessian)[i].block<3, 3>(miidx[j], 3 * i) += (-2.0 / mnorms[i]) * dnij[j].transpose();

                Eigen::Matrix3d term3 = (dnij[j].transpose() * mvec[i]) * oppNormals[i].transpose();

                (*hessian)[i].block<3, 3>(miidx[j], 3 * ip1) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term3;
                (*hessian)[i].block<3, 3>(miidx[j], 3 * ip2) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term3;
                (*hessian)[i].block<3, 3>(miidx[j], 3 * i) += (2.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term3;

                Eigen::Matrix3d term4 = (dcn.block<3, 3>(0, 3 * j).transpose() * mvec[i]) * oppNormals[i].transpose();

                (*hessian)[i].block<3, 3>(3 * j, 3 * ip1) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term4;
                (*hessian)[i].block<3, 3>(3 * j, 3 * ip2) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term4;
                (*hessian)[i].block<3, 3>(3 * j, 3 * i) += (2.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term4;

                (*hessian)[i].block<3, 3>(3 * ip1, miidx[j]) += (1.0 / mnorms[i]) * dnij[j];
                (*hessian)[i].block<3, 3>(3 * ip2, miidx[j]) += (1.0 / mnorms[i]) * dnij[j];
                (*hessian)[i].block<3, 3>(3 * i, miidx[j]) += (-2.0 / mnorms[i]) * dnij[j];

                for (int k = 0; k < 3; k++)
                {
                    (*hessian)[i].block<3, 3>(miidx[j], miidx[k]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * (dnij[j].transpose() * mvec[i]) * (qvec[i].transpose() * dnij[k]);
                    (*hessian)[i].block<3, 3>(3 * j, miidx[k]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * (dcn.block<3, 3>(0, 3 * j).transpose() * mvec[i]) * (qvec[i].transpose() * dnij[k]);
                }

                Eigen::Vector3d dnijm = dnij[j] * mvec[i];

                Eigen::Matrix3d term1 = oppNormals[i] * dnijm.transpose();
                (*hessian)[i].block<3, 3>(3 * ip1, miidx[j]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term1;
                (*hessian)[i].block<3, 3>(3 * ip2, miidx[j]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term1;
                (*hessian)[i].block<3, 3>(3 * i, miidx[j]) += (2.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term1;

                Eigen::Vector3d dcnjm = dcn.block<3, 3>(0, 3 * j) * mvec[i];
                
                Eigen::Matrix3d term2 = oppNormals[i] * dcnjm.transpose();
                (*hessian)[i].block<3, 3>(3 * ip1, 3 * j) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term2;
                (*hessian)[i].block<3, 3>(3 * ip2, 3 * j) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term2;
                (*hessian)[i].block<3, 3>(3 * i, 3 * j) += (2.0 / mnorms[i] / mnorms[i] / mnorms[i]) * term2;

                Eigen::Vector3d dnijTq = dnij[j].transpose() * qvec[i];

                for (int k = 0; k < 3; k++)
                {
                    (*hessian)[i].block<3, 3>(miidx[j], miidx[k]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * dnijTq * (mvec[i].transpose() * dnij[k]);
                    (*hessian)[i].block<3, 3>(miidx[j], 3 * k) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * dnijTq * (mvec[i].transpose() * dcn.block<3, 3>(0, 3 * k));
                }

                double qdoto = qvec[i].dot(oppNormals[i]);
                Eigen::Vector3d dnijTm = dnij[j].transpose() * mvec[i];
                Eigen::Vector3d dcnjTm = (dcn.block<3, 3>(0, 3 * j).transpose() * mvec[i]);
                for (int k = 0; k < 3; k++)
                {
                    (*hessian)[i].block<3, 3>(miidx[j], miidx[k]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dnij[j].transpose() * dnij[k];
                    (*hessian)[i].block<3, 3>(miidx[j], 3 * k) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dnij[j].transpose() * dcn.block<3, 3>(0, 3 * k);
                    (*hessian)[i].block<3, 3>(3 * j, miidx[k]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dcn.block<3, 3>(0, 3 * j).transpose() * dnij[k];
                    (*hessian)[i].block<3, 3>(3 * j, 3 * k) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dcn.block<3, 3>(0, 3 * j).transpose() * dcn.block<3, 3>(0, 3 * k);

                    (*hessian)[i].block<3, 3>(miidx[j], miidx[k]) += (3.0 / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dnijTm * (mvec[i].transpose() * dnij[k]);
                    (*hessian)[i].block<3, 3>(miidx[j], 3 * k) += (3.0 / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dnijTm * (mvec[i].transpose() * dcn.block<3, 3>(0, 3 * k));
                    (*hessian)[i].block<3, 3>(3 * j, miidx[k]) += (3.0 / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dcnjTm * (mvec[i].transpose() * dnij[k]);
                    (*hessian)[i].block<3, 3>(3 * j, 3 * k) += (3.0 / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * dcnjTm * (mvec[i].transpose() * dcn.block<3, 3>(0, 3 * k));

                    for (int l = 0; l < 3; l++)
                    {
                        (*hessian)[i].block<3,3>(miidx[j], miidx[k]) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * mvec[i][l] * hn[i][l].block<3,3>(3 * j, 3 * k);
                        (*hessian)[i].block<3, 3>(3 * j, 3 * k) += (-1.0 / mnorms[i] / mnorms[i] / mnorms[i]) * qdoto * mvec[i][l] * hcn[l].block<3, 3>(3 * j, 3 * k);
                        (*hessian)[i].block<3, 3>(miidx[j], miidx[k]) += (1.0 / mnorms[i]) * qvec[i][l] * hn[i][l].block<3, 3>(3 * j, 3 * k);
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
    Eigen::Matrix<double, 4, 18> *derivative, 
    std::vector<Eigen::Matrix<double, 18, 18> > *hessian)
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

constexpr int MidedgeAverageFormulation::numExtraDOFs;

void MidedgeAverageFormulation::initializeExtraDOFs(Eigen::VectorXd &extraDOFs, const MeshConnectivity &mesh, const Eigen::MatrixXd &curPos)
{
    extraDOFs.resize(0);
}