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
        const MonolayerRestState& rs = (const MonolayerRestState&)restState;

        constexpr int nedgedofs = SFF::numExtraDOFs;
        Matrix2d abarinv = rs.abars[face].inverse();
        Matrix<double, 4, 18 + 3 * nedgedofs> bderiv;
        std::vector<Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> > bhess;
        Matrix2d b = SFF::secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL, hessian ? &bhess : NULL);

        Matrix<double, 4, 9> aderivsmall;
        std::vector<Matrix<double, 9, 9> > ahesssmall;
        Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderivsmall : NULL, hessian ? &ahesssmall : NULL);
        Matrix<double, 4, 18 + 3 * nedgedofs> aderiv;
        if (derivative || hessian)
        {
            aderiv.setZero();
            aderiv.block(0, 0, 4, 9) = aderivsmall;
        }
        std::vector<Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> > ahess = bhess;
        if (hessian)
        {
            for (int i = 0; i < 4; i++)
            {
                ahess[i].setZero();
                ahess[i].block(0, 0, 9, 9) = ahesssmall[i];
            }
        }

        Matrix2d abaradj = adjugate(rs.abars[face]);
        Matrix2d bbaradj = adjugate(rs.bbars[face]);
        Matrix2d aadj = adjugate(a);
        Matrix2d badj = adjugate(b);
        double deta = a.determinant();
        double detabar = rs.abars[face].determinant();

        double coeff = std::sqrt(detabar) * pow(rs.thicknesses[face], 3) / 24;

        Matrix2d M = aadj * b / deta - abaradj * rs.bbars[face] / detabar;
        double result = coeff * (lameBeta_ * (M * M).trace() + 0.5 * lameAlpha_ * pow(M.trace(), 2));

        if (derivative)
        {
            derivative->setZero();

            double term1 = lameBeta_ * 2.0 / pow(deta, 2);

            Matrix2d m1 = aadj * b * aadj;
            *derivative += term1 * m1(0, 0) * bderiv.row(0);
            *derivative += term1 * m1(0, 1) * bderiv.row(1);
            *derivative += term1 * m1(1, 0) * bderiv.row(2);
            *derivative += term1 * m1(1, 1) * bderiv.row(3);

            Matrix2d m2 = badj * a * badj;
            *derivative += term1 * m2(0, 0) * aderiv.row(0);
            *derivative += term1 * m2(0, 1) * aderiv.row(1);
            *derivative += term1 * m2(1, 0) * aderiv.row(2);
            *derivative += term1 * m2(1, 1) * aderiv.row(3);

            double term2 = lameBeta_ * -2.0 / pow(deta, 3) * (aadj * b * aadj * b).trace();
            *derivative += term2 * aadj(0, 0) * aderiv.row(0);
            *derivative += term2 * aadj(0, 1) * aderiv.row(1);
            *derivative += term2 * aadj(1, 0) * aderiv.row(2);
            *derivative += term2 * aadj(1, 1) * aderiv.row(3);

            double term3 = lameBeta_ * -2.0 / deta;
            Matrix2d m3 = aadj * rs.bbars[face] * abaradj / detabar;
            *derivative += term3 * m3(0, 0) * bderiv.row(0);
            *derivative += term3 * m3(0, 1) * bderiv.row(1);
            *derivative += term3 * m3(1, 0) * bderiv.row(2);
            *derivative += term3 * m3(1, 1) * bderiv.row(3);

            Matrix2d m4 = badj * rs.abars[face] * bbaradj / detabar;
            *derivative += term3 * m4(0, 0) * aderiv.row(0);
            *derivative += term3 * m4(0, 1) * aderiv.row(1);
            *derivative += term3 * m4(1, 0) * aderiv.row(2);
            *derivative += term3 * m4(1, 1) * aderiv.row(3);

            double term4 = lameBeta_ * 2.0 / pow(deta, 2) * (aadj * b * abaradj * rs.bbars[face]).trace() / detabar;
            *derivative += term4 * aadj(0, 0) * aderiv.row(0);
            *derivative += term4 * aadj(0, 1) * aderiv.row(1);
            *derivative += term4 * aadj(1, 0) * aderiv.row(2);
            *derivative += term4 * aadj(1, 1) * aderiv.row(3);

            // end term 1

            double term5 = lameAlpha_ * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() * -1.0 / deta;
            *derivative += term5 * badj(0, 0) * aderiv.row(0);
            *derivative += term5 * badj(0, 1) * aderiv.row(1);
            *derivative += term5 * badj(1, 0) * aderiv.row(2);
            *derivative += term5 * badj(1, 1) * aderiv.row(3);

            *derivative += term5 * aadj(0, 0) * bderiv.row(0);
            *derivative += term5 * aadj(0, 1) * bderiv.row(1);
            *derivative += term5 * aadj(1, 0) * bderiv.row(2);
            *derivative += term5 * aadj(1, 1) * bderiv.row(3);

            double term6 = lameAlpha_ * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() * 1.0 / pow(deta, 2) * (aadj * b).trace();
            *derivative += term6 * aadj(0, 0) * aderiv.row(0);
            *derivative += term6 * aadj(0, 1) * aderiv.row(1);
            *derivative += term6 * aadj(1, 0) * aderiv.row(2);
            *derivative += term6 * aadj(1, 1) * aderiv.row(3);

            *derivative *= coeff;
        }

        if (hessian)
        {
            hessian->setZero();
            Matrix<double, 1, 18 + 3 * nedgedofs> aadjda = aadj(0, 0) * aderiv.row(0);
            aadjda += aadj(0, 1) * aderiv.row(1);
            aadjda += aadj(1, 0) * aderiv.row(2);
            aadjda += aadj(1, 1) * aderiv.row(3);

            double term1 = lameBeta_ * 2.0 / pow(deta, 2);
            Matrix2d m1 = aadj * b * aadj;
            *hessian += term1 * m1(0, 0) * bhess[0];
            *hessian += term1 * m1(0, 1) * bhess[1];
            *hessian += term1 * m1(1, 0) * bhess[2];
            *hessian += term1 * m1(1, 1) * bhess[3];
            Matrix2d m2 = aadj * b;
            *hessian += term1 * (m2(0, 0) * aderiv.row(3).transpose() + m2(0, 1) * -aderiv.row(2).transpose()) * bderiv.row(0);
            *hessian += term1 * (m2(0, 0) * -aderiv.row(1).transpose() + m2(0, 1) * aderiv.row(0).transpose()) * bderiv.row(1);
            *hessian += term1 * (m2(1, 0) * aderiv.row(3).transpose() + m2(1, 1) * -aderiv.row(2).transpose()) * bderiv.row(2);
            *hessian += term1 * (m2(1, 0) * -aderiv.row(1).transpose() + m2(1, 1) * aderiv.row(0).transpose()) * bderiv.row(3);

            *hessian += term1 * (aadj(0, 0) * bderiv.row(0).transpose() + aadj(0, 1) * bderiv.row(2).transpose()) * (aadj(0, 0) * bderiv.row(0) + aadj(0, 1) * bderiv.row(2));
            *hessian += term1 * (aadj(1, 0) * bderiv.row(0).transpose() + aadj(1, 1) * bderiv.row(2).transpose()) * (aadj(0, 0) * bderiv.row(1) + aadj(0, 1) * bderiv.row(3));
            *hessian += term1 * (aadj(0, 0) * bderiv.row(1).transpose() + aadj(0, 1) * bderiv.row(3).transpose()) * (aadj(1, 0) * bderiv.row(0) + aadj(1, 1) * bderiv.row(2));
            *hessian += term1 * (aadj(1, 0) * bderiv.row(1).transpose() + aadj(1, 1) * bderiv.row(3).transpose()) * (aadj(1, 0) * bderiv.row(1) + aadj(1, 1) * bderiv.row(3));

            Matrix2d m3 = a * badj;
            *hessian += term1 * (m3(1, 0) * aderiv.row(1).transpose() + m3(1, 1) * aderiv.row(3).transpose()) * bderiv.row(0);
            *hessian += term1 * -(m3(0, 0) * aderiv.row(1).transpose() + m3(0, 1) * aderiv.row(3).transpose()) * bderiv.row(1);
            *hessian += term1 * -(m3(1, 0) * aderiv.row(0).transpose() + m3(1, 1) * aderiv.row(2).transpose()) * bderiv.row(2);
            *hessian += term1 * (m3(0, 0) * aderiv.row(0).transpose() + m3(0, 1) * aderiv.row(2).transpose()) * bderiv.row(3);

            double term2 = lameBeta_ * -4.0 / pow(deta, 3);
            Matrix2d m4 = aadj * b * aadj;

            Matrix<double, 1, 18 + 3 * nedgedofs> m4db = m4(0, 0) * bderiv.row(0);
            m4db += m4(0, 1) * bderiv.row(1);
            m4db += m4(1, 0) * bderiv.row(2);
            m4db += m4(1, 1) * bderiv.row(3);
            *hessian += term2 * aadjda.transpose() * m4db;

            double term3 = lameBeta_ * 2.0 / pow(deta, 2);
            Matrix2d m5 = badj * a * badj;
            *hessian += term3 * m5(0, 0) * ahess[0];
            *hessian += term3 * m5(0, 1) * ahess[1];
            *hessian += term3 * m5(1, 0) * ahess[2];
            *hessian += term3 * m5(1, 1) * ahess[3];

            Matrix2d m6 = badj * a;
            *hessian += term3 * (m6(0, 0) * bderiv.row(3).transpose() + m6(0, 1) * -bderiv.row(2).transpose()) * aderiv.row(0);
            *hessian += term3 * (m6(0, 0) * -bderiv.row(1).transpose() + m6(0, 1) * bderiv.row(0).transpose()) * aderiv.row(1);
            *hessian += term3 * (m6(1, 0) * bderiv.row(3).transpose() + m6(1, 1) * -bderiv.row(2).transpose()) * aderiv.row(2);
            *hessian += term3 * (m6(1, 0) * -bderiv.row(1).transpose() + m6(1, 1) * bderiv.row(0).transpose()) * aderiv.row(3);

            *hessian += term3 * (badj(0, 0) * aderiv.row(0).transpose() + badj(0, 1) * aderiv.row(2).transpose()) * (badj(0, 0) * aderiv.row(0) + badj(0, 1) * aderiv.row(2));
            *hessian += term3 * (badj(1, 0) * aderiv.row(0).transpose() + badj(1, 1) * aderiv.row(2).transpose()) * (badj(0, 0) * aderiv.row(1) + badj(0, 1) * aderiv.row(3));
            *hessian += term3 * (badj(0, 0) * aderiv.row(1).transpose() + badj(0, 1) * aderiv.row(3).transpose()) * (badj(1, 0) * aderiv.row(0) + badj(1, 1) * aderiv.row(2));
            *hessian += term3 * (badj(1, 0) * aderiv.row(1).transpose() + badj(1, 1) * aderiv.row(3).transpose()) * (badj(1, 0) * aderiv.row(1) + badj(1, 1) * aderiv.row(3));

            Matrix2d m7 = b * aadj;
            *hessian += term3 * (m7(1, 0) * bderiv.row(1).transpose() + m7(1, 1) * bderiv.row(3).transpose()) * aderiv.row(0);
            *hessian += term3 * -(m7(0, 0) * bderiv.row(1).transpose() + m7(0, 1) * bderiv.row(3).transpose()) * aderiv.row(1);
            *hessian += term3 * -(m7(1, 0) * bderiv.row(0).transpose() + m7(1, 1) * bderiv.row(2).transpose()) * aderiv.row(2);
            *hessian += term3 * (m7(0, 0) * bderiv.row(0).transpose() + m7(0, 1) * bderiv.row(2).transpose()) * aderiv.row(3);

            double term4 = lameBeta_ * -4.0 / pow(deta, 3);
            Matrix2d m8 = badj * a * badj;
            Matrix<double, 1, 18 + 3 * nedgedofs> m8da = m8(0, 0) * aderiv.row(0);
            m8da += m8(0, 1) * aderiv.row(1);
            m8da += m8(1, 0) * aderiv.row(2);
            m8da += m8(1, 1) * aderiv.row(3);
            *hessian += term4 * aadjda.transpose() * m8da;

            double term5 = lameBeta_ * -2.0 / pow(deta, 3) * (aadj * b * aadj * b).trace();
            *hessian += term5 * aadj(0, 0) * ahess[0];
            *hessian += term5 * aadj(0, 1) * ahess[1];
            *hessian += term5 * aadj(1, 0) * ahess[2];
            *hessian += term5 * aadj(1, 1) * ahess[3];

            *hessian += term5 * aderiv.row(3).transpose() * aderiv.row(0);
            *hessian += term5 * -aderiv.row(1).transpose() * aderiv.row(1);
            *hessian += term5 * -aderiv.row(2).transpose() * aderiv.row(2);
            *hessian += term5 * aderiv.row(0).transpose() * aderiv.row(3);

            double term6 = lameBeta_ * -4.0 / pow(deta, 3);
            Matrix2d m9 = aadj * b * aadj;
            Matrix<double, 1, 18 + 3 * nedgedofs> m9db = m9(0, 0) * bderiv.row(0);
            m9db += m9(0, 1) * bderiv.row(1);
            m9db += m9(1, 0) * bderiv.row(2);
            m9db += m9(1, 1) * bderiv.row(3);
            *hessian += term6 * m9db.transpose() * aadjda;

            Matrix2d m10 = badj * a * badj;
            Matrix<double, 1, 18 + 3 * nedgedofs> m10da = m10(0, 0) * aderiv.row(0);
            m10da += m10(0, 1) * aderiv.row(1);
            m10da += m10(1, 0) * aderiv.row(2);
            m10da += m10(1, 1) * aderiv.row(3);
            *hessian += term6 * m10da.transpose() * aadjda;

            double term7 = lameBeta_ * 6.0 / pow(deta, 4) * (aadj * b * aadj * b).trace();
            *hessian += term7 * aadjda.transpose() * aadjda;

            double term8 = lameBeta_ * -2.0 / deta;
            Matrix2d m11 = rs.abars[face] * bbaradj / detabar;
            *hessian += term8 * (m11(1, 0) * aderiv.row(1).transpose() + m11(1, 1) * aderiv.row(3).transpose()) * bderiv.row(0);
            *hessian += term8 * -(m11(0, 0) * aderiv.row(1).transpose() + m11(0, 1) * aderiv.row(3).transpose()) * bderiv.row(1);
            *hessian += term8 * -(m11(1, 0) * aderiv.row(0).transpose() + m11(1, 1) * aderiv.row(2).transpose()) * bderiv.row(2);
            *hessian += term8 * (m11(0, 0) * aderiv.row(0).transpose() + m11(0, 1) * aderiv.row(2).transpose()) * bderiv.row(3);

            Matrix2d m12 = aadj * rs.bbars[face] * abaradj / detabar;
            *hessian += term8 * m12(0, 0) * bhess[0];
            *hessian += term8 * m12(0, 1) * bhess[1];
            *hessian += term8 * m12(1, 0) * bhess[2];
            *hessian += term8 * m12(1, 1) * bhess[3];

            double term9 = lameBeta_ * 2.0 / pow(deta, 2);
            Matrix2d m13 = aadj * rs.bbars[face] * abaradj / detabar;
            Matrix<double, 1, 18 + 3 * nedgedofs> m13db = m13(0, 0) * bderiv.row(0);
            m13db += m13(0, 1) * bderiv.row(1);
            m13db += m13(1, 0) * bderiv.row(2);
            m13db += m13(1, 1) * bderiv.row(3);

            *hessian += term9 * aadjda.transpose() * m13db;

            double term10 = lameBeta_ * -2.0 / deta;
            Matrix2d m14 = rs.bbars[face] * abaradj / detabar;
            *hessian += term10 * (m14(1, 0) * bderiv.row(1).transpose() + m14(1, 1) * bderiv.row(3).transpose()) * aderiv.row(0);
            *hessian += term10 * -(m14(0, 0) * bderiv.row(1).transpose() + m14(0, 1) * bderiv.row(3).transpose()) * aderiv.row(1);
            *hessian += term10 * -(m14(1, 0) * bderiv.row(0).transpose() + m14(1, 1) * bderiv.row(2).transpose()) * aderiv.row(2);
            *hessian += term10 * (m14(0, 0) * bderiv.row(0).transpose() + m14(0, 1) * bderiv.row(2).transpose()) * aderiv.row(3);

            Matrix2d m15 = badj * rs.abars[face] * bbaradj / detabar;
            *hessian += term10 * m15(0, 0) * ahess[0];
            *hessian += term10 * m15(0, 1) * ahess[1];
            *hessian += term10 * m15(1, 0) * ahess[2];
            *hessian += term10 * m15(1, 1) * ahess[3];

            double term11 = lameBeta_ * 2.0 / pow(deta, 2);
            Matrix2d m16 = badj * rs.abars[face] * bbaradj / detabar;
            Matrix<double, 1, 18 + 3 * nedgedofs> m16da = m16(0, 0) * aderiv.row(0);
            m16da += m16(0, 1) * aderiv.row(1);
            m16da += m16(1, 0) * aderiv.row(2);
            m16da += m16(1, 1) * aderiv.row(3);

            *hessian += term11 * aadjda.transpose() * m16da;

            *hessian += term11 * m16da.transpose() * aadjda;
            *hessian += term11 * m13db.transpose() * aadjda;

            double term12 = lameBeta_ * 2.0 / pow(deta, 2) * (aadj * b * abaradj * rs.bbars[face]).trace() / detabar;
            *hessian += term12 * aderiv.row(3).transpose() * aderiv.row(0);
            *hessian += term12 * -aderiv.row(1).transpose() * aderiv.row(1);
            *hessian += term12 * -aderiv.row(2).transpose() * aderiv.row(2);
            *hessian += term12 * aderiv.row(0).transpose() * aderiv.row(3);
            *hessian += term12 * aadj(0, 0) * ahess[0];
            *hessian += term12 * aadj(0, 1) * ahess[1];
            *hessian += term12 * aadj(1, 0) * ahess[2];
            *hessian += term12 * aadj(1, 1) * ahess[3];

            double term13 = lameBeta_ * -4.0 / pow(deta, 2) / deta * (aadj * b * abaradj * rs.bbars[face]).trace() / detabar;
            *hessian += term13 * aadjda.transpose() * aadjda;

            // end term 1

            double term14 = lameAlpha_ * -1.0 * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() / deta;
            *hessian += term14 * bderiv.row(3).transpose() * aderiv.row(0);
            *hessian += term14 * -bderiv.row(1).transpose() * aderiv.row(1);
            *hessian += term14 * -bderiv.row(2).transpose() * aderiv.row(2);
            *hessian += term14 * bderiv.row(0).transpose() * aderiv.row(3);
            *hessian += term14 * badj(0, 0) * ahess[0];
            *hessian += term14 * badj(0, 1) * ahess[1];
            *hessian += term14 * badj(1, 0) * ahess[2];
            *hessian += term14 * badj(1, 1) * ahess[3];
            *hessian += term14 * aderiv.row(3).transpose() * bderiv.row(0);
            *hessian += term14 * -aderiv.row(1).transpose() * bderiv.row(1);
            *hessian += term14 * -aderiv.row(2).transpose() * bderiv.row(2);
            *hessian += term14 * aderiv.row(0).transpose() * bderiv.row(3);
            *hessian += term14 * aadj(0, 0) * bhess[0];
            *hessian += term14 * aadj(0, 1) * bhess[1];
            *hessian += term14 * aadj(1, 0) * bhess[2];
            *hessian += term14 * aadj(1, 1) * bhess[3];

            double term15 = lameAlpha_ * 1.0 * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() / pow(deta, 2);
            Matrix<double, 1, 18 + 3 * nedgedofs> badjda = badj(0, 0) * aderiv.row(0);
            badjda += badj(0, 1) * aderiv.row(1);
            badjda += badj(1, 0) * aderiv.row(2);
            badjda += badj(1, 1) * aderiv.row(3);
            *hessian += term15 * aadjda.transpose() * badjda;
            Matrix<double, 1, 18 + 3 * nedgedofs> aadjdb = aadj(0, 0) * bderiv.row(0);
            aadjdb += aadj(0, 1) * bderiv.row(1);
            aadjdb += aadj(1, 0) * bderiv.row(2);
            aadjdb += aadj(1, 1) * bderiv.row(3);
            *hessian += term15 * aadjda.transpose() * aadjdb;

            double term16 = lameAlpha_ * 1.0 * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() / pow(deta, 2) * (aadj * b).trace();
            *hessian += term16 * aadj(0, 0) * ahess[0];
            *hessian += term16 * aadj(1, 0) * ahess[1];
            *hessian += term16 * aadj(0, 1) * ahess[2];
            *hessian += term16 * aadj(1, 1) * ahess[3];
            *hessian += term16 * aderiv.row(3).transpose() * aderiv.row(0);
            *hessian += term16 * -aderiv.row(1).transpose() * aderiv.row(1);
            *hessian += term16 * -aderiv.row(2).transpose() * aderiv.row(2);
            *hessian += term16 * aderiv.row(0).transpose() * aderiv.row(3);

            double term17 = lameAlpha_ * 1.0 * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() / pow(deta, 2);
            *hessian += term17 * aadjdb.transpose() * aadjda;
            *hessian += term17 * badjda.transpose() * aadjda;

            double term18 = lameAlpha_ * -2.0 * (abaradj * rs.bbars[face] / detabar - aadj * b / deta).trace() / pow(deta, 3) * (aadj * b).trace();
            *hessian += term18 * aadjda.transpose() * aadjda;

            double term19 = lameAlpha_ / pow(deta, 2);
            *hessian += term19 * badjda.transpose() * badjda;
            *hessian += term19 * badjda.transpose() * aadjdb;
            *hessian += term19 * aadjdb.transpose() * badjda;
            *hessian += term19 * aadjdb.transpose() * aadjdb;

            double term20 = lameAlpha_ * -1.0 / pow(deta, 3) * (aadj * b).trace();
            *hessian += term20 * aadjda.transpose() * badjda;
            *hessian += term20 * aadjda.transpose() * aadjdb;
            *hessian += term20 * badjda.transpose() * aadjda;
            *hessian += term20 * aadjdb.transpose() * aadjda;

            double term21 = lameAlpha_ / pow(deta, 4) * (aadj * b).trace() * (aadj * b).trace();
            *hessian += term21 * aadjda.transpose() * aadjda;

            *hessian *= coeff;
        }

        return result;
    }

    // instantiations
    template class NeoHookeanMaterial<MidedgeAngleSinFormulation>;
    template class NeoHookeanMaterial<MidedgeAngleTanFormulation>;
    template class NeoHookeanMaterial<MidedgeAverageFormulation>;

};