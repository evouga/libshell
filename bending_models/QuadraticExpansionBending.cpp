#include "QuadraticExpansionBending.h"
#include "../include/MeshConnectivity.h"
#include "../include/RestState.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAverageFormulation.h"

template <class SFF>
void bendingMatrixTerm(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& restExtraDOFs,
    const LibShell::RestState& restState,    
    int face,
    Eigen::Matrix<double, 18 + 3 * SFF::numExtraDOFs, 18 + 3 * SFF::numExtraDOFs>& matrixTerm)
{
    using namespace Eigen;

    assert(restState.type() == LibShell::RestStateType::RST_MONOLAYER);
    const LibShell::MonolayerRestState& rs = (const LibShell::MonolayerRestState&)restState;

    double coeff = pow(rs.thicknesses[face], 3) / 12;
    constexpr int nedgedofs = SFF::numExtraDOFs;
    Matrix2d abarinv = rs.abars[face].inverse();
    Matrix<double, 4, 18 + 3 * nedgedofs> bderiv;
    Matrix2d b = SFF::secondFundamentalForm(mesh, restPos, restExtraDOFs, face, &bderiv, NULL);
    double dA = 0.5 * sqrt(rs.abars[face].determinant());

    Matrix<double, 1, 18 + 3 * nedgedofs> inner = bderiv.transpose() * Map<Vector4d>(abarinv.data());
    double lameAlpha = ((LibShell::MonolayerRestState&)restState).lameAlpha[face];
    double lameBeta = ((LibShell::MonolayerRestState&)restState).lameBeta[face];

    matrixTerm = lameAlpha * inner.transpose() * inner;

    Matrix<double, 1, 18 + 3 * nedgedofs> inner00 = abarinv(0, 0) * bderiv.row(0) + abarinv(0, 1) * bderiv.row(2);
    Matrix<double, 1, 18 + 3 * nedgedofs> inner01 = abarinv(0, 0) * bderiv.row(1) + abarinv(0, 1) * bderiv.row(3);
    Matrix<double, 1, 18 + 3 * nedgedofs> inner10 = abarinv(1, 0) * bderiv.row(0) + abarinv(1, 1) * bderiv.row(2);
    Matrix<double, 1, 18 + 3 * nedgedofs> inner11 = abarinv(1, 0) * bderiv.row(1) + abarinv(1, 1) * bderiv.row(3);
    matrixTerm += 2 * lameBeta * inner00.transpose() * inner00;
    matrixTerm += 2 * lameBeta * (inner01.transpose() * inner10 + inner10.transpose() * inner01);
    matrixTerm += 2 * lameBeta * inner11.transpose() * inner11;

    matrixTerm *= coeff * dA;

}

template <class SFF>
void bendingMatrix(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& restExtraDOFs,
    const LibShell::RestState& restState,    
    std::vector<Eigen::Triplet<double> >& Mcoeffs
)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)restPos.rows();

    constexpr int nedgedofs = SFF::numExtraDOFs;
    
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> hess;
        bendingMatrixTerm<SFF>(mesh, restPos, restExtraDOFs, restState, i, hess);
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    for (int m = 0; m < 3; m++)
                    {
                        Mcoeffs.push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * mesh.faceVertex(i, k) + m, hess(3 * j + l, 3 * k + m)));
                        int oppidxk = mesh.vertexOppositeFaceEdge(i, k);
                        if (oppidxk != -1)
                            Mcoeffs.push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * oppidxk + m, hess(3 * j + l, 9 + 3 * k + m)));
                        int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                        if (oppidxj != -1)
                            Mcoeffs.push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * mesh.faceVertex(i, k) + m, hess(9 + 3 * j + l, 3 * k + m)));
                        if (oppidxj != -1 && oppidxk != -1)
                            Mcoeffs.push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * oppidxk + m, hess(9 + 3 * j + l, 9 + 3 * k + m)));
                    }
                    for (int m = 0; m < nedgedofs; m++)
                    {
                        Mcoeffs.push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hess(3 * j + l, 18 + nedgedofs * k + m)));
                        Mcoeffs.push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * mesh.faceVertex(i, j) + l, hess(18 + nedgedofs * k + m, 3 * j + l)));
                        int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                        if (oppidxj != -1)
                        {
                            Mcoeffs.push_back(Eigen::Triplet<double>(3 * oppidxj + l, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, hess(9 + 3 * j + l, 18 + nedgedofs * k + m)));
                            Mcoeffs.push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, k) + m, 3 * oppidxj + l, hess(18 + nedgedofs * k + m, 9 + 3 * j + l)));
                        }
                    }
                }
                for (int m = 0; m < nedgedofs; m++)
                {
                    for (int n = 0; n < nedgedofs; n++)
                    {
                        Mcoeffs.push_back(Eigen::Triplet<double>(3 * nverts + nedgedofs * mesh.faceEdge(i, j) + m, 3 * nverts + nedgedofs * mesh.faceEdge(i, k) + n, hess(18 + nedgedofs * j + m, 18 + nedgedofs * k + n)));
                    }
                }
            }
        }
    }
}

template void bendingMatrix<LibShell::MidedgeAngleSinFormulation>(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& restExtraDOFs,
    const LibShell::RestState& restState,
    std::vector<Eigen::Triplet<double> >& Mcoeffs
);

template void bendingMatrix<LibShell::MidedgeAngleTanFormulation>(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& restExtraDOFs,
    const LibShell::RestState& restState,
    std::vector<Eigen::Triplet<double> >& Mcoeffs
    );

template void bendingMatrix<LibShell::MidedgeAverageFormulation>(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::VectorXd& restExtraDOFs,
    const LibShell::RestState& restState,
    std::vector<Eigen::Triplet<double> >& Mcoeffs
    );
