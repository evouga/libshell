#include "../../include/StVKMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include "../../include/MidedgeAngleSinFormulation.h"
#include "../../include/MidedgeAngleTanFormulation.h"
#include "../../include/MidedgeAverageFormulation.h"
#include "../../include/MidedgeAngleThetaFormulation.h"
#include "../../include/RestState.h"

namespace LibShell {

    template <class SFF>
    double StVKMaterial<SFF>::stretchingEnergy(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const RestState& restState,
        int face,
        Eigen::Matrix<double, 1, 9>* derivative, // F(face, i)
        Eigen::Matrix<double, 9, 9>* hessian) const
    {
        using namespace Eigen;

        assert(restState.type() == RestStateType::RST_MONOLAYER);
        const MonolayerRestState& rs = (const MonolayerRestState&)restState;

        double coeff = rs.thicknesses[face] / 4.0;
        Matrix2d abarinv = rs.abars[face].inverse();
        Matrix<double, 4, 9> aderiv;
        std::vector<Matrix<double, 9, 9> > ahess;
        Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);
        Matrix2d M = abarinv * (a - rs.abars[face]);
        double dA = 0.5 * sqrt(rs.abars[face].determinant());
        double lameAlpha = rs.lameAlpha[face];
        double lameBeta = rs.lameBeta[face];

        double StVK = 0.5 * lameAlpha * pow(M.trace(), 2) + lameBeta * (M * M).trace();
        double result = coeff * dA * StVK;

        if (derivative)
        {
            Matrix2d temp = lameAlpha * M.trace() * abarinv + 2 * lameBeta * M * abarinv;
            *derivative = coeff * dA * aderiv.transpose() * Map<Vector4d>(temp.data());
        }

        if (hessian)
        {
            Matrix<double, 1, 9> inner = aderiv.transpose() * Map<Vector4d>(abarinv.data());
            *hessian = lameAlpha * inner.transpose() * inner;

            Matrix2d Mainv = M * abarinv;
            for (int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
                *hessian += (lameAlpha * M.trace() * abarinv(i) + 2 * lameBeta * Mainv(i)) * ahess[i];

            Matrix<double, 1, 9> inner00 = abarinv(0, 0) * aderiv.row(0) + abarinv(0, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner01 = abarinv(0, 0) * aderiv.row(1) + abarinv(0, 1) * aderiv.row(3);
            Matrix<double, 1, 9> inner10 = abarinv(1, 0) * aderiv.row(0) + abarinv(1, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner11 = abarinv(1, 0) * aderiv.row(1) + abarinv(1, 1) * aderiv.row(3);
            *hessian += 2 * lameBeta * inner00.transpose() * inner00;
            *hessian += 2 * lameBeta * (inner01.transpose() * inner10 + inner10.transpose() * inner01);
            *hessian += 2 * lameBeta * inner11.transpose() * inner11;

            *hessian *= coeff * dA;
        }

        return result;
    }

    template <class SFF>
    double StVKMaterial<SFF>::bendingEnergy(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        const RestState& restState,
        int face,
        Eigen::Matrix<double, 1, 18 + 3 * SFF::numExtraDOFs>* derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
        Eigen::Matrix<double, 18 + 3 * SFF::numExtraDOFs, 18 + 3 * SFF::numExtraDOFs>* hessian) const
    {
        using namespace Eigen;

        assert(restState.type() == RestStateType::RST_MONOLAYER);
        const MonolayerRestState& rs = (const MonolayerRestState&)restState;

        double coeff = pow(rs.thicknesses[face], 3) / 12;
        constexpr int nedgedofs = SFF::numExtraDOFs;
        Matrix2d abarinv = rs.abars[face].inverse();
        Matrix<double, 4, 18 + 3 * nedgedofs> bderiv;
        std::vector<Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> > bhess;
        Matrix2d b = SFF::secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL, hessian ? &bhess : NULL);
        Matrix2d M = abarinv * (b - rs.bbars[face]);
        double dA = 0.5 * sqrt(rs.abars[face].determinant());
        double lameAlpha = rs.lameAlpha[face];
        double lameBeta = rs.lameBeta[face];

        double StVK = 0.5 * lameAlpha * pow(M.trace(), 2) + lameBeta * (M * M).trace();
        double result = coeff * dA * StVK;

        if (derivative)
        {
            Matrix2d temp = lameAlpha * M.trace() * abarinv + 2 * lameBeta * M * abarinv;
            *derivative = coeff * dA * bderiv.transpose() * Map<Vector4d>(temp.data());
        }

        if (hessian)
        {
            Matrix<double, 1, 18 + 3 * nedgedofs> inner = bderiv.transpose() * Map<Vector4d>(abarinv.data());
            *hessian = lameAlpha * inner.transpose() * inner;

            Matrix2d Mainv = M * abarinv;
            for (int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
                *hessian += (lameAlpha * M.trace() * abarinv(i) + 2 * lameBeta * Mainv(i)) * bhess[i];

            Matrix<double, 1, 18 + 3 * nedgedofs> inner00 = abarinv(0, 0) * bderiv.row(0) + abarinv(0, 1) * bderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner01 = abarinv(0, 0) * bderiv.row(1) + abarinv(0, 1) * bderiv.row(3);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner10 = abarinv(1, 0) * bderiv.row(0) + abarinv(1, 1) * bderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner11 = abarinv(1, 0) * bderiv.row(1) + abarinv(1, 1) * bderiv.row(3);
            *hessian += 2 * lameBeta * inner00.transpose() * inner00;
            *hessian += 2 * lameBeta * (inner01.transpose() * inner10 + inner10.transpose() * inner01);
            *hessian += 2 * lameBeta * inner11.transpose() * inner11;

            *hessian *= coeff * dA;
        }

        return result;
    }

    // instantiations
    template class StVKMaterial<MidedgeAngleSinFormulation>;
    template class StVKMaterial<MidedgeAngleTanFormulation>;
    template class StVKMaterial<MidedgeAverageFormulation>;
    template class StVKMaterial<MidedgeAngleThetaFormulation>;
};