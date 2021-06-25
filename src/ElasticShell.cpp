#include "libshell/ElasticShell.h"
#include "libshell/MaterialModel.h"
#include "libshell/MeshConnectivity.h"
#include "libshell/MidedgeAngleSinFormulation.h"
#include "libshell/MidedgeAngleTanFormulation.h"
#include "libshell/MidedgeAverageFormulation.h"
#include "libshell/RestState.h"
#include "GeometryDerivatives.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace LibShell {

    template <class SFF>
    double ElasticShell<SFF>::elasticEnergy(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        const MaterialModel<SFF>& mat,
        const RestState& restState,
        Eigen::VectorXd* derivative, // positions, then thetas
        std::vector<Eigen::Triplet<double> >* hessian)
    {
        return elasticEnergy(mesh, curPos, extraDOFs, mat, restState,
            EnergyTerm::ET_BENDING | EnergyTerm::ET_STRETCHING,
            derivative, hessian);
    }

    template <class SFF>
    double ElasticShell<SFF>::elasticEnergy(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        const MaterialModel<SFF>& mat,
        const RestState& restState,
        int whichTerms,
        Eigen::VectorXd* derivative, // positions, then thetas
        std::vector<Eigen::Triplet<double> >* hessian)
    {
        int nfaces = mesh.nFaces();
        int nedges = mesh.nEdges();
        int nverts = (int)curPos.rows();

        if (curPos.cols() != 3 || extraDOFs.size() != SFF::numExtraDOFs * nedges)
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
        if (whichTerms & EnergyTerm::ET_STRETCHING)
        {
            for (int i = 0; i < nfaces; i++)
            {
                Eigen::Matrix<double, 1, 9> deriv;
                Eigen::Matrix<double, 9, 9> hess;
                result += mat.stretchingEnergy(mesh, curPos, restState, i, derivative ? &deriv : NULL, hessian ? &hess : NULL);
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
        }

        // bending terms
        if (whichTerms & EnergyTerm::ET_BENDING)
        {
            constexpr int nedgedofs = SFF::numExtraDOFs;
            for (int i = 0; i < nfaces; i++)
            {
                Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> deriv;
                Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> hess;
                result += mat.bendingEnergy(mesh, curPos, extraDOFs, restState, i, derivative ? &deriv : NULL, hessian ? &hess : NULL);
                if (derivative)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        derivative->segment<3>(3 * mesh.faceVertex(i, j)) += deriv.template block<1, 3>(0, 3 * j).transpose();
                        int oppidx = mesh.vertexOppositeFaceEdge(i, j);
                        if (oppidx != -1)
                            derivative->segment<3>(3 * oppidx) += deriv.template block<1, 3>(0, 9 + 3 * j).transpose();
                        for (int k = 0; k < nedgedofs; k++)
                        {
                            (*derivative)[3 * nverts + nedgedofs * mesh.faceEdge(i, j) + k] += deriv(0, 18 + nedgedofs * j + k);
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
                                    if (oppidxk != -1)
                                        hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * oppidxk + m, hess(3 * j + l, 9 + 3 * k + m)));
                                    int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                                    if (oppidxj != -1)
                                        hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * mesh.faceVertex(i, k) + m, hess(9 + 3 * j + l, 3 * k + m)));
                                    if (oppidxj != -1 && oppidxk != -1)
                                        hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * oppidxk + m, hess(9 + 3 * j + l, 9 + 3 * k + m)));
                                }
                                for (int m = 0; m < nedgedofs; m++)
                                {
                                    hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hess(3 * j + l, 18 + nedgedofs * k + m)));
                                    hessian->push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * mesh.faceVertex(i, j) + l, hess(18 + nedgedofs * k + m, 3 * j + l)));
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
        }
        return result;
    }

    template <class SFF>
    void ElasticShell<SFF>::firstFundamentalForms(const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos, std::vector<Eigen::Matrix2d>& abars)
    {
        int nfaces = mesh.nFaces();
        abars.resize(nfaces);
        for (int i = 0; i < nfaces; i++)
        {
            abars[i] = firstFundamentalForm(mesh, curPos, i, NULL, NULL);
        }
    }

    template <class SFF>
    void ElasticShell<SFF>::secondFundamentalForms(const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos, const Eigen::VectorXd& edgeDOFs, std::vector<Eigen::Matrix2d>& bbars)
    {
        int nfaces = mesh.nFaces();
        bbars.resize(nfaces);
        for (int i = 0; i < nfaces; i++)
        {
            bbars[i] = SFF::secondFundamentalForm(mesh, curPos, edgeDOFs, i, NULL, NULL);
        }
    }

    // instantions
    template class ElasticShell<MidedgeAngleSinFormulation>;
    template class ElasticShell<MidedgeAngleTanFormulation>;
    template class ElasticShell<MidedgeAverageFormulation>;

}