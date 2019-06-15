#include "../include/ElasticShell.h"
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <iostream>
#include <Eigen/Sparse>
#include "GeometryDerivatives.h"
#include <random>
#include <iostream>
#include <vector>
#include "../include/MeshConnectivity.h"
#include "../include/MaterialModel.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAverageFormulation.h"

template <class SFF>
double ElasticShell<SFF>::elasticEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &extraDOFs,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars, 
    const std::vector<Eigen::Matrix2d> &bbars,
    Eigen::VectorXd *derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> > *hessian)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    if (curPos.cols() != 3 || extraDOFs.size() != SFF::numExtraDOFs * nedges || thicknesses.size() != nfaces || abars.size() != nfaces || bbars.size() != nfaces)
    {
        return std::numeric_limits<double>::infinity();
    }

    if (derivative)
    {
        derivative->resize(3 * nverts + SFF::numExtraDOFs * nedges);
        derivative->setZero();
    }
    if (hessian)
    {
        hessian->clear();
    }

    double result = 0;
    
    // stretching terms
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Matrix<double, 1, 9> deriv;
        Eigen::Matrix<double, 9, 9> hess;
        result += mat.stretchingEnergy(mesh, curPos, thicknesses[i], abars[i], i, derivative ? &deriv : NULL, hessian ? &hess : NULL);
        if (derivative)
        {
            for (int j = 0; j < 3; j++)
                derivative->segment<3>(3 * mesh.faceVertex(i, j)) += deriv.segment<3>(3 * j);
        }
        if (hessian)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        for (int m = 0; m < 3; m++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * mesh.faceVertex(i, k) + m, hess(3 * j + l, 3 * k + m)));
                        }
                    }
                }
            }
        }
    }
    
    
    // bending terms
    constexpr int nedgedofs = SFF::numExtraDOFs;
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> deriv;
        Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> hess;
        result += mat.bendingEnergy(mesh, curPos, extraDOFs, thicknesses[i], abars[i], bbars[i], i, derivative ? &deriv : NULL, hessian ? &hess : NULL);
        if (derivative)
        {
            for (int j = 0; j < 3; j++)
            {
                derivative->segment<3>(3 * mesh.faceVertex(i, j)) += deriv.block<1,3>(0, 3 * j).transpose();
                int oppidx = mesh.vertexOppositeFaceEdge(i, j);
                if(oppidx != -1)
                    derivative->segment<3>(3 * oppidx) += deriv.block<1,3>(0, 9 + 3 * j).transpose();
                for (int k = 0; k < nedgedofs; k++)
                {
                    (*derivative)[3 * nverts + nedgedofs * mesh.faceEdge(i, j) + k] += deriv(0, 18 + nedgedofs *j + k);
                }
            }
        }
        if (hessian)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++)
                    {
                        for (int m = 0; m < 3; m++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * mesh.faceVertex(i, k) + m, hess(3 * j + l, 3 * k + m)));
                            int oppidxk = mesh.vertexOppositeFaceEdge(i, k);
                            if(oppidxk != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * oppidxk + m, hess(3 * j + l, 9 + 3 * k + m)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if(oppidxj != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * mesh.faceVertex(i, k) + m, hess(9 + 3 * j + l, 3 * k + m)));
                            if(oppidxj != -1 && oppidxk != -1)
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * oppidxk + m, hess(9 + 3 * j + l, 9 + 3 * k + m)));
                        }
                        for (int m = 0; m < nedgedofs; m++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hess(3 * j + l, 18 + nedgedofs*k + m)));
                            hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * mesh.faceVertex(i, j) + l, hess(18 + nedgedofs*k + m, 3 * j + l)));
                            int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                            if (oppidxj != -1)
                            {
                                hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hess(9 + 3 * j + l, 18 + nedgedofs * k + m)));
                                hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * oppidxj + l, hess(18 + nedgedofs * k + m, 9 + 3 * j + l)));
                            }
                        }
                    }
                    for (int m = 0; m < nedgedofs; m++)
                    {
                        for (int n = 0; n < nedgedofs; n++)
                        {
                            hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, j) + m, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + n, hess(18 + nedgedofs * j + m, 18 + nedgedofs * k + n)));
                        }
                    }
                }
            }
        }
    }
    return result;
}

template <class SFF>
void ElasticShell<SFF>::firstFundamentalForms(const MeshConnectivity &mesh, const Eigen::MatrixXd &curPos, std::vector<Eigen::Matrix2d> &abars)
{
    int nfaces = mesh.nFaces();
    abars.resize(nfaces);
    for (int i = 0; i < nfaces; i++)
    {
        abars[i] = firstFundamentalForm(mesh, curPos, i, NULL, NULL);
    }
}

template <class SFF>
void ElasticShell<SFF>::secondFundamentalForms(const MeshConnectivity &mesh, const Eigen::MatrixXd &curPos, const Eigen::VectorXd &edgeDOFs, std::vector<Eigen::Matrix2d> &bbars)
{
    int nfaces = mesh.nFaces();
    bbars.resize(nfaces);
    for (int i = 0; i < nfaces; i++)
    {
        bbars[i] = SFF::secondFundamentalForm(mesh, curPos, edgeDOFs, i, NULL, NULL);
    }
}

template <class SFF>
void ElasticShell<SFF>::testStretchingFiniteDifferences(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    Eigen::MatrixXd testpos = curPos;
    testpos.setRandom();
    
    int numtests = 100;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> facegen(0,nfaces-1);
    double pert = 1e-6;
    
    for (int i = 0; i < numtests; i++)
    {

        int face = facegen(rng);
        std::cout << "Face " << face << std::endl;
        Eigen::Matrix<double, 1, 9> deriv;
        Eigen::Matrix<double, 9, 9> hess;
        double result = mat.stretchingEnergy(mesh, testpos, thicknesses[face], abars[face], face, &deriv, &hess);

        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                Eigen::MatrixXd pertpos = testpos;
                pertpos(mesh.faceVertex(face, j), k) += pert;
                Eigen::Matrix<double, 1, 9> pertderiv;
                double newresult = mat.stretchingEnergy(mesh, pertpos, thicknesses[face], abars[face], face, &pertderiv, NULL);
                double findiff = (newresult - result) / pert;
                std::cout << '(' << j << ", " << k << ") " << findiff << " " << deriv(0, 3 * j + k) << std::endl;
                Eigen::Matrix<double, 1, 9> derivdiff = (pertderiv - deriv) / pert;
                std::cout << derivdiff << std::endl;
                std::cout << "//" << std::endl;
                std::cout << hess.row(3 * j + k) << std::endl << std::endl;
            }
        }
    } 
}

template <class SFF>
void ElasticShell<SFF>::testBendingFiniteDifferences(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &edgeDOFs,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    const std::vector<Eigen::Matrix2d> &bbars)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    Eigen::MatrixXd testpos = curPos;
    testpos.setRandom();
    Eigen::VectorXd testedge = edgeDOFs;
    testedge.setRandom();
    
    int numtests = 100;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> facegen(0,nfaces-1);
    double pert = 1e-6;
    constexpr int nedgedofs = SFF::numExtraDOFs;

    for (int i = 0; i < numtests; i++)
    {
        
        int face = facegen(rng);
        std::cout << "Face " << face << std::endl;
        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> deriv;
        Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> hess;
        double result = mat.bendingEnergy(mesh, testpos, testedge, thicknesses[face], abars[face], bbars[face], face, &deriv, &hess);

        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                Eigen::MatrixXd pertpos = testpos;
                pertpos(mesh.faceVertex(face, j), k) += pert;
                Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> pertderiv;
                double newresult = mat.bendingEnergy(mesh, pertpos, testedge, thicknesses[face], abars[face], bbars[face], face, &pertderiv, NULL);
                double findiff = (newresult - result) / pert;
                std::cout << '(' << j << ", " << k << ") " << findiff << " " << deriv(0, 3 * j + k) << std::endl;
                Eigen::MatrixXd derivdiff = (pertderiv - deriv) / pert;
                std::cout << derivdiff << std::endl;
                std::cout << "//" << std::endl;
                std::cout << hess.row(3 * j + k) << std::endl << std::endl;                                
            }
            int oppidx = mesh.vertexOppositeFaceEdge(face, j);
            if (oppidx != -1)
            {
                for (int k = 0; k < 3; k++)
                {
                    Eigen::MatrixXd pertpos = testpos;
                    pertpos(oppidx, k) += pert;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> pertderiv;
                    double newresult = mat.bendingEnergy(mesh, pertpos, testedge, thicknesses[face], abars[face], bbars[face], face, &pertderiv, NULL);
                    double findiff = (newresult - result) / pert;
                    Eigen::MatrixXd derivdiff = (pertderiv - deriv) / pert;

                    std::cout << "opp (" << j << ", " << k << ") " << findiff << " " << deriv(0, 9 + 3 * j + k) << std::endl;
                    std::cout << derivdiff << std::endl;
                    std::cout << "//" << std::endl;
                    std::cout << hess.row(9 + 3 * j + k) << std::endl << std::endl;
                }
            }
            for (int k = 0; k < nedgedofs; k++)
            {
                Eigen::VectorXd pertedge = testedge;
                pertedge[nedgedofs * mesh.faceEdge(face, j) + k] += pert;
                Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> pertderiv;
                double newresult = mat.bendingEnergy(mesh, testpos, pertedge, thicknesses[face], abars[face], bbars[face], face, &pertderiv, NULL);
                double findiff = (newresult - result) / pert;
                std::cout << findiff << " " << deriv(0, 18 + nedgedofs * j + k) << std::endl;
                Eigen::MatrixXd derivdiff = (pertderiv - deriv) / pert;
                std::cout << derivdiff << std::endl;
                std::cout << "//" << std::endl;
                std::cout << hess.row(18 + nedgedofs * j + k) << std::endl << std::endl;
            }            
        }
    } 
}

// instantions
template class ElasticShell<MidedgeAngleSinFormulation>;
template class ElasticShell<MidedgeAngleTanFormulation>;
template class ElasticShell<MidedgeAverageFormulation>;