#include "../../include/NeoHookeanMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include <iostream>
#include "../../include/MidedgeAngleSinFormulation.h"
#include "../../include/MidedgeAngleTanFormulation.h"
#include "../../include/MidedgeAverageFormulation.h"
#include "../../include/RestState.h"

namespace LibShell {

    template <class SFF>
    double NeoHookeanMaterial<SFF>::stretchingEnergy(
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

        Matrix<double, 4, 9> aderiv;
        std::vector<Matrix<double, 9, 9> > ahess;
        Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);

        double deta = a.determinant();
        double detabar = rs.abars[face].determinant();
        double lnJ = std::log(deta / detabar) / 2;
        Matrix2d abarinv = adjugate(rs.abars[face]) / detabar;

        double result = lameBeta_ * ((abarinv * a).trace() - 2 - 2 * lnJ) + lameAlpha_ * pow(lnJ, 2);
        double coeff = rs.thicknesses[face] * std::sqrt(detabar) / 4;
        result *= coeff;

        if (derivative)
        {
            Matrix2d ainv = adjugate(a) / deta;

            Matrix2d temp = lameBeta_ * abarinv + (-lameBeta_ + lameAlpha_ * lnJ) * ainv;

            *derivative = aderiv.transpose() * Map<Vector4d>(temp.data());
            *derivative *= coeff;
        }

        if (hessian)
        {
            hessian->setZero();

            Matrix2d ainv = adjugate(a) / deta;
            double term1 = -lameBeta_ + lameAlpha_ * lnJ;

            Matrix<double, 1, 9> ainvda = aderiv.transpose() * Map<Vector4d>(ainv.data());
            *hessian = (-term1 + lameAlpha_ / 2) * ainvda.transpose() * ainvda;

            Matrix<double, 4, 9> aderivadj;
            aderivadj << aderiv.row(3), -aderiv.row(1), -aderiv.row(2), aderiv.row(0);

            *hessian += term1 / deta * aderivadj.transpose() * aderiv;

            for (int i = 0; i < 4; ++i)
                *hessian += (term1 * ainv(i) + lameBeta_ * abarinv(i)) * ahess[i];

            *hessian *= coeff;
        }

        return result;
    }

    template <class SFF>
    double NeoHookeanMaterial<SFF>::bendingEnergy(
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
        const MonolayerRestState &rs = (const MonolayerRestState &)restState;

        constexpr int nedgedofs = SFF::numExtraDOFs;
        Matrix<double, 4, 18 + 3 * nedgedofs> bderiv;
        std::vector<Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs>> bhess;
        Matrix2d b = SFF::secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL,
                                                hessian ? &bhess : NULL);

        double detabar = rs.abars[face].determinant();
        Matrix2d abarinv = adjugate(rs.abars[face]) / detabar;

        double coeff = std::sqrt(detabar) * pow(rs.thicknesses[face], 3) / 24;

        Matrix2d M = abarinv * (b - rs.bbars[face]);
        if((M + Matrix2d::Identity()).determinant() < 0)
        {
            if(derivative)
            derivative->setZero();
            if(hessian)
            hessian->setZero();
            return 0;
        }

        double lnJ = std::log((M + Matrix2d::Identity()).determinant()) / 2;
        double result = coeff * (lameBeta_ * (M.trace() - 2 * lnJ) + lameAlpha_ * pow(lnJ, 2));

        if(derivative || hessian)
        {
            Matrix2d binv = (b + rs.abars[face] - rs.bbars[face]).inverse();
            Matrix2d temp = (-lameBeta_ + lameAlpha_ * lnJ) * binv + lameBeta_ * abarinv;

            if(derivative)
            *derivative = coeff * bderiv.transpose() * Map<Vector4d>(temp.data());

            if(hessian)
            {
            Matrix<double, 1, 18 + 3 *nedgedofs> binvdb = bderiv.transpose() * Map<Vector4d>(binv.data());
            *hessian = (lameBeta_ - lameAlpha_ * lnJ + lameAlpha_ / 2) * binvdb.transpose() * binvdb;

            Matrix<double, 4, 18 + 3 * nedgedofs> bderivadj;
            bderivadj << bderiv.row(3), -bderiv.row(1), -bderiv.row(2), bderiv.row(0);

            *hessian += (-lameBeta_ + lameAlpha_ * lnJ) * binv.determinant() * bderivadj.transpose() * bderiv;

            for(int i = 0; i < 4; ++i)
                *hessian += temp(i) * bhess[i];

            *hessian *= coeff;
            }
        }
        return result;
    }

    // instantiations
    template class NeoHookeanMaterial<MidedgeAngleSinFormulation>;
    template class NeoHookeanMaterial<MidedgeAngleTanFormulation>;
    template class NeoHookeanMaterial<MidedgeAverageFormulation>;

};