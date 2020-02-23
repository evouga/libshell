#include "../../include/StVKMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include "../../include/MidedgeAngleSinFormulation.h"
#include "../../include/MidedgeAngleTanFormulation.h"
#include "../../include/MidedgeAverageFormulation.h"

template <class SFF>
double StVKMaterial<SFF>::stretchingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    double thickness,
    const Eigen::Matrix2d &abar,
    int face,
    Eigen::Matrix<double, 1, 9> *derivative, // F(face, i)
    Eigen::Matrix<double, 9, 9> *hessian) const
{
    using namespace Eigen;

    double coeff = thickness / 4.0;
    Matrix2d abarinv = abar.inverse();
    Matrix<double, 4, 9> aderiv;
    std::vector<Matrix<double, 9, 9> > ahess;
    Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);
    Matrix2d M = abarinv * (a - abar);
    double dA = 0.5 * sqrt(abar.determinant());

    double StVK = 0.5 * lameAlpha_ * pow(M.trace(), 2) + lameBeta_ * (M*M).trace();
    double result = coeff * dA * StVK;

    if (derivative)
    {
        Matrix2d temp = lameAlpha_ * M.trace() * abarinv + 2 * lameBeta_ * M * abarinv;
        *derivative = coeff * dA * aderiv.transpose() * Map<Vector4d>(temp.data());
    }

    if (hessian)
    {
        Matrix<double, 1, 9> inner = aderiv.transpose() * Map<Vector4d>(abarinv.data());
        *hessian = lameAlpha_ * inner.transpose() * inner;

        Matrix2d Mainv = M*abarinv;
        for(int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
            *hessian += (lameAlpha_ * M.trace() * abarinv(i) + 2 * lameBeta_ * Mainv(i)) * ahess[i];

        Matrix<double, 1, 9> inner00 = abarinv(0, 0) * aderiv.row(0) + abarinv(0, 1) * aderiv.row(2);
        Matrix<double, 1, 9> inner01 = abarinv(0, 0) * aderiv.row(1) + abarinv(0, 1) * aderiv.row(3);
        Matrix<double, 1, 9> inner10 = abarinv(1, 0) * aderiv.row(0) + abarinv(1, 1) * aderiv.row(2);
        Matrix<double, 1, 9> inner11 = abarinv(1, 0) * aderiv.row(1) + abarinv(1, 1) * aderiv.row(3);
        *hessian += 2 * lameBeta_ * inner00.transpose() * inner00;
        *hessian += 2 * lameBeta_ * (inner01.transpose() * inner10  + inner10.transpose() * inner01);
        *hessian += 2 * lameBeta_ * inner11.transpose() * inner11;

        *hessian *= coeff * dA;
    }

    return result;
}

template <class SFF>
double StVKMaterial<SFF>::bendingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &extraDOFs,
    double thickness,
    const Eigen::Matrix2d &abar, const Eigen::Matrix2d &bbar,
    int face,
    Eigen::Matrix<double, 1, 18 + 3*SFF::numExtraDOFs> *derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
    Eigen::Matrix<double, 18 + 3*SFF::numExtraDOFs, 18 + 3*SFF::numExtraDOFs> *hessian) const
{
    using namespace Eigen;

    double coeff = pow(thickness, 3) / 12;
    constexpr int nedgedofs = SFF::numExtraDOFs;
    Matrix2d abarinv = abar.inverse();
    Matrix<double, 4, 18 + 3*nedgedofs> bderiv;
    std::vector<Matrix<double, 18 + 3*nedgedofs, 18 + 3*nedgedofs> > bhess;
    Matrix2d b = SFF::secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL, hessian ? &bhess : NULL);
    Matrix2d M = abarinv * (b - bbar);
    double dA = 0.5 * sqrt(abar.determinant());

    double StVK = 0.5 * lameAlpha_ * pow(M.trace(), 2) + lameBeta_ * (M*M).trace();
    double result = coeff * dA * StVK;

    if (derivative)
    {
        Matrix2d temp = lameAlpha_ * M.trace() * abarinv + 2 * lameBeta_ * M * abarinv;
        *derivative = coeff * dA * bderiv.transpose() * Map<Vector4d>(temp.data());
    }

    if (hessian)
    {
        Matrix<double, 1, 18 + 3*nedgedofs> inner = bderiv.transpose() * Map<Vector4d>(abarinv.data());
        *hessian = lameAlpha_ * inner.transpose() * inner;

        Matrix2d Mainv = M*abarinv;
        for(int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
            *hessian += (lameAlpha_ * M.trace() * abarinv(i) + 2 * lameBeta_ * Mainv(i)) * bhess[i];

        Matrix<double, 1, 18 + 3*nedgedofs> inner00 = abarinv(0, 0) * bderiv.row(0) + abarinv(0, 1) * bderiv.row(2);
        Matrix<double, 1, 18 + 3*nedgedofs> inner01 = abarinv(0, 0) * bderiv.row(1) + abarinv(0, 1) * bderiv.row(3);
        Matrix<double, 1, 18 + 3*nedgedofs> inner10 = abarinv(1, 0) * bderiv.row(0) + abarinv(1, 1) * bderiv.row(2);
        Matrix<double, 1, 18 + 3*nedgedofs> inner11 = abarinv(1, 0) * bderiv.row(1) + abarinv(1, 1) * bderiv.row(3);
        *hessian += 2 * lameBeta_ * inner00.transpose() * inner00;
        *hessian += 2 * lameBeta_ * (inner01.transpose() * inner10  + inner10.transpose() * inner01);
        *hessian += 2 * lameBeta_ * inner11.transpose() * inner11;

        *hessian *= coeff * dA;
    }

    return result;
}

// instantiations
template class StVKMaterial<MidedgeAngleSinFormulation>;
template class StVKMaterial<MidedgeAngleTanFormulation>;
template class StVKMaterial<MidedgeAverageFormulation>;
