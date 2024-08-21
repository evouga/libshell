#include "../../include/BilayerStVKMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include "../../include/MidedgeAngleSinFormulation.h"
#include "../../include/MidedgeAngleTanFormulation.h"
#include "../../include/MidedgeAverageFormulation.h"
#include "../../include/MidedgeAngleThetaFormulation.h"
#include "../../include/RestState.h"
#include <iostream>

namespace LibShell {

    template <class SFF>
    double BilayerStVKMaterial<SFF>::stretchingEnergy(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const RestState& restState,
        int face,
        Eigen::Matrix<double, 1, 9>* derivative, // F(face, i)
        Eigen::Matrix<double, 9, 9>* hessian) const
    {
        using namespace Eigen;

        assert(restState.type() == RestStateType::RST_BILAYER);
        const BilayerRestState& rs = (const BilayerRestState&)restState;

        Matrix<double, 4, 9> aderiv;
        std::vector<Matrix<double, 9, 9> > ahess;
        Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);

        double coeff1 = rs.layers[0].thicknesses[face] / 8.0;
        Matrix2d abar1inv = rs.layers[0].abars[face].inverse();
        Matrix2d M1 = abar1inv * (a - rs.layers[0].abars[face]);
        double dA1 = 0.5 * sqrt(rs.layers[0].abars[face].determinant());
        double lameAlpha1 = rs.layers[0].lameAlpha[face];
        double lameBeta1 = rs.layers[0].lameBeta[face];

        double coeff2 = rs.layers[1].thicknesses[face] / 8.0;
        Matrix2d abar2inv = rs.layers[1].abars[face].inverse();
        Matrix2d M2 = abar2inv * (a - rs.layers[1].abars[face]);
        double dA2 = 0.5 * sqrt(rs.layers[1].abars[face].determinant());
        double lameAlpha2 = rs.layers[1].lameAlpha[face];
        double lameBeta2 = rs.layers[1].lameBeta[face];

        double StVK1 = 0.5 * lameAlpha1 * pow(M1.trace(), 2) + lameBeta1 * (M1 * M1).trace();
        double StVK2 = 0.5 * lameAlpha2 * pow(M2.trace(), 2) + lameBeta2 * (M2 * M2).trace();
        double result = coeff1 * dA1 * StVK1 + coeff2 * dA2 * StVK2;

        if (derivative)
        {
            Matrix2d temp1 = lameAlpha1 * M1.trace() * abar1inv + 2 * lameBeta1 * M1 * abar1inv;
            Matrix2d temp2 = lameAlpha2 * M2.trace() * abar2inv + 2 * lameBeta2 * M2 * abar2inv;
            *derivative = coeff1 * dA1 * aderiv.transpose() * Map<Vector4d>(temp1.data());
            *derivative += coeff2 * dA2 * aderiv.transpose() * Map<Vector4d>(temp2.data());
        }

        if (hessian)
        {
            Matrix<double, 1, 9> inner1 = aderiv.transpose() * Map<Vector4d>(abar1inv.data());
            Matrix<double, 1, 9> inner2 = aderiv.transpose() * Map<Vector4d>(abar2inv.data());
            *hessian = coeff1 * dA1 * lameAlpha1 * inner1.transpose() * inner1;
            *hessian += coeff2 * dA2 * lameAlpha2 * inner2.transpose() * inner2;

            Matrix2d Mainv1 = M1 * abar1inv;
            Matrix2d Mainv2 = M2 * abar2inv;
            for (int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
            {
                *hessian += coeff1 * dA1 * (lameAlpha1 * M1.trace() * abar1inv(i) + 2 * lameBeta1 * Mainv1(i)) * ahess[i];
                *hessian += coeff2 * dA2 * (lameAlpha2 * M2.trace() * abar2inv(i) + 2 * lameBeta2 * Mainv2(i)) * ahess[i];
            }

            Matrix<double, 1, 9> inner001 = abar1inv(0, 0) * aderiv.row(0) + abar1inv(0, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner011 = abar1inv(0, 0) * aderiv.row(1) + abar1inv(0, 1) * aderiv.row(3);
            Matrix<double, 1, 9> inner101 = abar1inv(1, 0) * aderiv.row(0) + abar1inv(1, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner111 = abar1inv(1, 0) * aderiv.row(1) + abar1inv(1, 1) * aderiv.row(3);
            *hessian += 2 * coeff1 * dA1 * lameBeta1 * inner001.transpose() * inner001;
            *hessian += 2 * coeff1 * dA1 * lameBeta1 * (inner011.transpose() * inner101 + inner101.transpose() * inner011);
            *hessian += 2 * coeff1 * dA1 * lameBeta1 * inner111.transpose() * inner111;

            Matrix<double, 1, 9> inner002 = abar2inv(0, 0) * aderiv.row(0) + abar2inv(0, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner012 = abar2inv(0, 0) * aderiv.row(1) + abar2inv(0, 1) * aderiv.row(3);
            Matrix<double, 1, 9> inner102 = abar2inv(1, 0) * aderiv.row(0) + abar2inv(1, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner112 = abar2inv(1, 0) * aderiv.row(1) + abar2inv(1, 1) * aderiv.row(3);
            *hessian += 2 * coeff2 * dA2 * lameBeta2 * inner002.transpose() * inner002;
            *hessian += 2 * coeff2 * dA2 * lameBeta2 * (inner012.transpose() * inner102 + inner102.transpose() * inner012);
            *hessian += 2 * coeff2 * dA2 * lameBeta2 * inner112.transpose() * inner112;
        }

        return result;
    }

    template <class SFF>
    double BilayerStVKMaterial<SFF>::bendingEnergy(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        const RestState& restState,
        int face,
        Eigen::Matrix<double, 1, 18 + 3 * SFF::numExtraDOFs>* derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
        Eigen::Matrix<double, 18 + 3 * SFF::numExtraDOFs, 18 + 3 * SFF::numExtraDOFs>* hessian) const
    {
        using namespace Eigen;

        assert(restState.type() == RestStateType::RST_BILAYER);
        const BilayerRestState& rs = (const BilayerRestState&)restState;

        constexpr int nedgedofs = SFF::numExtraDOFs;
        Matrix<double, 4, 18 + 3 * nedgedofs> bderiv;
        std::vector<Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> > bhess;
        Matrix2d b = SFF::secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL, hessian ? &bhess : NULL);

        double coeff1 = pow(rs.layers[0].thicknesses[face], 3) / 24;
        Matrix2d abarinv1 = rs.layers[0].abars[face].inverse();
        Matrix2d M1 = abarinv1 * (b - rs.layers[0].bbars[face]);
        double dA1 = 0.5 * sqrt(rs.layers[0].abars[face].determinant());
        double lameAlpha1 = rs.layers[0].lameAlpha[face];
        double lameBeta1 = rs.layers[0].lameBeta[face];

        double coeff2 = pow(rs.layers[1].thicknesses[face], 3) / 24;
        Matrix2d abarinv2 = rs.layers[1].abars[face].inverse();
        Matrix2d M2 = abarinv2 * (b - rs.layers[1].bbars[face]);
        double dA2 = 0.5 * sqrt(rs.layers[1].abars[face].determinant());
        double lameAlpha2 = rs.layers[1].lameAlpha[face];
        double lameBeta2 = rs.layers[1].lameBeta[face];

        double StVK1 = 0.5 * lameAlpha1 * pow(M1.trace(), 2) + lameBeta1 * (M1 * M1).trace();
        double StVK2 = 0.5 * lameAlpha2 * pow(M2.trace(), 2) + lameBeta2 * (M2 * M2).trace();        

        double result = coeff1 * dA1 * StVK1 + coeff2 * dA2 * StVK2;

        Matrix<double, 4, 9> aderiv;
        std::vector<Matrix<double, 9, 9> > ahess;
        Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);

        std::vector<Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> > augahess(4);
        if (hessian)
        {
            for (int i = 0; i < 4; i++)
            {
                augahess[i].setZero();
                augahess[i].template block<9, 9>(0, 0) = ahess[i];
            }
        }

        double crossTermCoeff1 = std::pow(rs.layers[0].thicknesses[face], 2) / 8.0;
        Matrix2d sigma1 = abarinv1 * (a - rs.layers[0].abars[face]);

        double crossTermCoeff2 = -std::pow(rs.layers[1].thicknesses[face], 2) / 8.0;
        Matrix2d sigma2 = abarinv2 * (a - rs.layers[1].abars[face]);

        double crossTerm1 = 0.5 * lameAlpha1 * sigma1.trace() * M1.trace() + lameBeta1 * (sigma1 * M1).trace();
        double crossTerm2 = 0.5 * lameAlpha2 * sigma2.trace() * M2.trace() + lameBeta2 * (sigma2 * M2).trace();
        result += crossTermCoeff1 * dA1 * crossTerm1 + crossTermCoeff2 * dA2 * crossTerm2;

        if (derivative)
        {
            Matrix2d temp1 = 0.5 * lameAlpha1 * M1.trace() * abarinv1 + lameBeta1 * M1 * abarinv1;
            Matrix2d temp2 = 0.5 * lameAlpha2 * M2.trace() * abarinv2 + lameBeta2 * M2 * abarinv2;
            *derivative = 2.0 * coeff1 * dA1 * bderiv.transpose() * Map<Vector4d>(temp1.data());
            *derivative += 2.0 * coeff2 * dA2 * bderiv.transpose() * Map<Vector4d>(temp2.data());

            derivative->template segment<9>(0) += crossTermCoeff1 * dA1 * aderiv.transpose() * Map<Vector4d>(temp1.data());
            derivative->template segment<9>(0) += crossTermCoeff2 * dA2 * aderiv.transpose() * Map<Vector4d>(temp2.data());

            Matrix2d temp3 = 0.5 * lameAlpha1 * sigma1.trace() * abarinv1 + lameBeta1 * sigma1 * abarinv1;
            Matrix2d temp4 = 0.5 * lameAlpha2 * sigma2.trace() * abarinv2 + lameBeta2 * sigma2 * abarinv2;
            *derivative += crossTermCoeff1 * dA1 * bderiv.transpose() * Map<Vector4d>(temp3.data());
            *derivative += crossTermCoeff2 * dA2 * bderiv.transpose() * Map<Vector4d>(temp4.data());
        }

        if (hessian)
        {
            Matrix<double, 1, 18 + 3 * nedgedofs> inner1 = bderiv.transpose() * Map<Vector4d>(abarinv1.data());
            *hessian = coeff1 * dA1 * lameAlpha1 * inner1.transpose() * inner1;

            Matrix<double, 1, 18 + 3 * nedgedofs> inner2 = bderiv.transpose() * Map<Vector4d>(abarinv2.data());
            *hessian += coeff2 * dA2 * lameAlpha2 * inner2.transpose() * inner2;

            Matrix<double, 1, 18 + 3 * nedgedofs> inner3;
            inner3.setZero();
            Matrix<double, 1, 18 + 3 * nedgedofs> inner4;
            inner4.setZero();

            inner3.template segment<9>(0) = aderiv.transpose() * Map<Vector4d>(abarinv1.data());
            inner4.template segment<9>(0) = aderiv.transpose() * Map<Vector4d>(abarinv2.data());

            *hessian += crossTermCoeff1 * dA1 * 0.5 * lameAlpha1 * inner1.transpose() * inner3;
            *hessian += crossTermCoeff1 * dA1 * 0.5 * lameAlpha1 * inner3.transpose() * inner1;

            *hessian += crossTermCoeff2 * dA2 * 0.5 * lameAlpha2 * inner2.transpose() * inner4;
            *hessian += crossTermCoeff2 * dA2 * 0.5 * lameAlpha2 * inner4.transpose() * inner2;

            Matrix2d Mainv1 = M1 * abarinv1;
            Matrix2d Mainv2 = M2 * abarinv2;

            Matrix2d Sainv1 = sigma1 * abarinv1;
            Matrix2d Sainv2 = sigma2 * abarinv2;

            for (int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
            {
                *hessian += coeff1 * dA1 * (lameAlpha1 * M1.trace() * abarinv1(i) + 2 * lameBeta1 * Mainv1(i)) * bhess[i];
                *hessian += coeff2 * dA2 * (lameAlpha2 * M2.trace() * abarinv2(i) + 2 * lameBeta2 * Mainv2(i)) * bhess[i];

                *hessian += crossTermCoeff1 * dA1 * (0.5 * lameAlpha1 * M1.trace() * abarinv1(i) + lameBeta1 * Mainv1(i)) * augahess[i];
                *hessian += crossTermCoeff2 * dA2 * (0.5 * lameAlpha2 * M2.trace() * abarinv2(i) + lameBeta2 * Mainv2(i)) * augahess[i];
                *hessian += crossTermCoeff1 * dA1 * (0.5 * lameAlpha1 * sigma1.trace() * abarinv1(i) + lameBeta1 * Sainv1(i)) * bhess[i];
                *hessian += crossTermCoeff2 * dA2 * (0.5 * lameAlpha2 * sigma2.trace() * abarinv2(i) + lameBeta2 * Sainv2(i)) * bhess[i];
            }

            Matrix<double, 1, 18 + 3 * nedgedofs> inner001 = abarinv1(0, 0) * bderiv.row(0) + abarinv1(0, 1) * bderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner011 = abarinv1(0, 0) * bderiv.row(1) + abarinv1(0, 1) * bderiv.row(3);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner101 = abarinv1(1, 0) * bderiv.row(0) + abarinv1(1, 1) * bderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner111 = abarinv1(1, 0) * bderiv.row(1) + abarinv1(1, 1) * bderiv.row(3);
            *hessian += 2 * coeff1 * dA1 * lameBeta1 * inner001.transpose() * inner001;
            *hessian += 2 * coeff1 * dA1 * lameBeta1 * (inner011.transpose() * inner101 + inner101.transpose() * inner011);
            *hessian += 2 * coeff1 * dA1 * lameBeta1 * inner111.transpose() * inner111;

            Matrix<double, 1, 18 + 3 * nedgedofs> inner002 = abarinv2(0, 0) * bderiv.row(0) + abarinv2(0, 1) * bderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner012 = abarinv2(0, 0) * bderiv.row(1) + abarinv2(0, 1) * bderiv.row(3);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner102 = abarinv2(1, 0) * bderiv.row(0) + abarinv2(1, 1) * bderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner112 = abarinv2(1, 0) * bderiv.row(1) + abarinv2(1, 1) * bderiv.row(3);
            *hessian += 2 * coeff2 * dA2 * lameBeta2 * inner002.transpose() * inner002;
            *hessian += 2 * coeff2 * dA2 * lameBeta2 * (inner012.transpose() * inner102 + inner102.transpose() * inner012);
            *hessian += 2 * coeff2 * dA2 * lameBeta2 * inner112.transpose() * inner112;

            Matrix<double, 1, 18 + 3 * nedgedofs> inner003;
            inner003.setZero();
            inner003.template segment<9>(0) = abarinv1(0, 0) * aderiv.row(0) + abarinv1(0, 1) * aderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner013;
            inner013.setZero();
            inner013.template segment<9>(0) = abarinv1(0, 0) * aderiv.row(1) + abarinv1(0, 1) * aderiv.row(3);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner103;
            inner103.setZero();
            inner103.template segment<9>(0) = abarinv1(1, 0) * aderiv.row(0) + abarinv1(1, 1) * aderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner113;
            inner113.setZero();
            inner113.template segment<9>(0) = abarinv1(1, 0) * aderiv.row(1) + abarinv1(1, 1) * aderiv.row(3);

            *hessian += crossTermCoeff1 * dA1 * lameBeta1 * inner001.transpose() * inner003;
            *hessian += crossTermCoeff1 * dA1 * lameBeta1 * inner003.transpose() * inner001;
            *hessian += crossTermCoeff1 * dA1 * lameBeta1 * (inner011.transpose() * inner103 + inner103.transpose() * inner011);
            *hessian += crossTermCoeff1 * dA1 * lameBeta1 * (inner013.transpose() * inner101 + inner101.transpose() * inner013);
            *hessian += crossTermCoeff1 * dA1 * lameBeta1 * inner111.transpose() * inner113;
            *hessian += crossTermCoeff1 * dA1 * lameBeta1 * inner113.transpose() * inner111;

            Matrix<double, 1, 18 + 3 * nedgedofs> inner004;
            inner004.setZero();
            inner004.template segment<9>(0) = abarinv2(0, 0) * aderiv.row(0) + abarinv2(0, 1) * aderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner014;
            inner014.setZero();
            inner014.template segment<9>(0) = abarinv2(0, 0) * aderiv.row(1) + abarinv2(0, 1) * aderiv.row(3);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner104;
            inner104.setZero();
            inner104.template segment<9>(0) = abarinv2(1, 0) * aderiv.row(0) + abarinv2(1, 1) * aderiv.row(2);
            Matrix<double, 1, 18 + 3 * nedgedofs> inner114;
            inner114.setZero();
            inner114.template segment<9>(0) = abarinv2(1, 0) * aderiv.row(1) + abarinv2(1, 1) * aderiv.row(3);

            *hessian += crossTermCoeff2 * dA2 * lameBeta2 * inner002.transpose() * inner004;
            *hessian += crossTermCoeff2 * dA2 * lameBeta2 * inner004.transpose() * inner002;
            *hessian += crossTermCoeff2 * dA2 * lameBeta2 * (inner012.transpose() * inner104 + inner104.transpose() * inner012);
            *hessian += crossTermCoeff2 * dA2 * lameBeta2 * (inner014.transpose() * inner102 + inner102.transpose() * inner014);
            *hessian += crossTermCoeff2 * dA2 * lameBeta2 * inner112.transpose() * inner114;
            *hessian += crossTermCoeff2 * dA2 * lameBeta2 * inner114.transpose() * inner112;
        }

        return result;
    }

    // instantiations
    template class BilayerStVKMaterial<MidedgeAngleSinFormulation>;
    template class BilayerStVKMaterial<MidedgeAngleTanFormulation>;
    template class BilayerStVKMaterial<MidedgeAverageFormulation>;
    template class BilayerStVKMaterial<MidedgeAngleThetaFormulation>;
};