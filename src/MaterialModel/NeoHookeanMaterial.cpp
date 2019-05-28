#include "../../include/NeoHookeanMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include "../../include/SecondFundamentalFormDiscretization.h"
#include <iostream>

double NeoHookeanMaterial::stretchingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,    
    double thickness,
    const Eigen::Matrix2d &abar,
    int face,
    Eigen::Matrix<double, 1, 9> *derivative, // F(face, i)
    Eigen::Matrix<double, 9, 9> *hessian) const
{
    double coeff1 = thickness * lameAlpha_ / 2.0;
    Eigen::Matrix2d abaradj = adjugate(abar);
    Eigen::Matrix<double, 4, 9> aderiv;
    std::vector<Eigen::Matrix<double, 9, 9> > ahess;
    Eigen::Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);
    double rtdeta = sqrt(a.determinant());
    double rtdetabar = sqrt(abar.determinant());
    double result = coeff1 * rtdetabar * ((abaradj * a).trace() / abar.determinant() - 2.0 - std::log(a.determinant()) + std::log(abar.determinant()));

    double coeff2 = thickness * lameBeta_ / 2.0;
    result += coeff2 * rtdetabar * 0.5 * (std::log(a.determinant()) - std::log(abar.determinant()))* (std::log(a.determinant()) - std::log(abar.determinant()));

    if (derivative)
    {
        derivative->setZero();
        Eigen::Matrix2d aadj = adjugate(a);
        
        double term1 = coeff1 / rtdetabar;

        *derivative += term1 * abaradj(0, 0) * aderiv.row(0);
        *derivative += term1 * abaradj(0, 1) * aderiv.row(1);
        *derivative += term1 * abaradj(1, 0) * aderiv.row(2);
        *derivative += term1 * abaradj(1, 1) * aderiv.row(3);

        double term2 = -coeff1 * rtdetabar / a.determinant();
        *derivative += term2 * aadj(0, 0) * aderiv.row(0);
        *derivative += term2 * aadj(0, 1) * aderiv.row(1);
        *derivative += term2 * aadj(1, 0) * aderiv.row(2);
        *derivative += term2 * aadj(1, 1) * aderiv.row(3);   

        double term3 = coeff2 * rtdetabar * (std::log(a.determinant()) - std::log(abar.determinant())) / a.determinant();
        *derivative += term3 * aadj(0, 0) * aderiv.row(0);
        *derivative += term3 * aadj(0, 1) * aderiv.row(1);
        *derivative += term3 * aadj(1, 0) * aderiv.row(2);
        *derivative += term3 * aadj(1, 1) * aderiv.row(3);
    }

    if (hessian)
    {
        hessian->setZero();
        
        Eigen::Matrix2d aadj = adjugate(a);
        
        Eigen::Matrix<double, 1, 9> aadjda = aadj(0, 0) * aderiv.row(0);
        aadjda += aadj(0, 1) * aderiv.row(1);
        aadjda += aadj(1, 0) * aderiv.row(2);
        aadjda += aadj(1, 1) * aderiv.row(3);

        double term1 = coeff1 / rtdetabar;
        *hessian += term1 * abaradj(0, 0) * ahess[0];
        *hessian += term1 * abaradj(0, 1) * ahess[1];
        *hessian += term1 * abaradj(1, 0) * ahess[2];
        *hessian += term1 * abaradj(1, 1) * ahess[3];

        double term2 = -coeff1 * rtdetabar / a.determinant();
        *hessian += term2 * aadj(0, 0) * ahess[0];
        *hessian += term2 * aadj(0, 1) * ahess[1];
        *hessian += term2 * aadj(1, 0) * ahess[2];
        *hessian += term2 * aadj(1, 1) * ahess[3];
        *hessian += term2 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term2 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term2 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term2 * aderiv.row(0).transpose() * aderiv.row(3);

        double term3 = coeff1 * rtdetabar / a.determinant() / a.determinant();
        *hessian += term3 * aadjda.transpose() * aadjda;

        double term4 = coeff2 * rtdetabar * (std::log(a.determinant()) - std::log(abar.determinant())) / a.determinant();
        *hessian += term4 * aadj(0, 0) * ahess[0];
        *hessian += term4 * aadj(0, 1) * ahess[1];
        *hessian += term4 * aadj(1, 0) * ahess[2];
        *hessian += term4 * aadj(1, 1) * ahess[3];

        *hessian += term4 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term4 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term4 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term4 * aderiv.row(0).transpose() * aderiv.row(3);

        double term5 = coeff2 * rtdetabar * -(std::log(a.determinant()) - std::log(abar.determinant())) / a.determinant() / a.determinant();
        *hessian += term5 * aadjda.transpose() * aadjda;

        double term6 = coeff2 * rtdetabar / a.determinant() / a.determinant();
        *hessian += term6 * aadjda.transpose() * aadjda;
    }

    return result;
}

double NeoHookeanMaterial::bendingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &extraDOFs,
    double thickness,
    const Eigen::Matrix2d &abar, const Eigen::Matrix2d &bbar,
    int face,
    const SecondFundamentalFormDiscretization &sff,
    Eigen::MatrixXd *derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
    Eigen::MatrixXd *hessian) const
{
    double coeff1 = thickness * thickness*thickness *lameAlpha_ / 24.0;
    double coeff2 = thickness * thickness*thickness * lameBeta_ / 24.0;
    int nedgedofs = sff.numExtraDOFs();
    Eigen::Matrix2d abarinv = abar.inverse();
    Eigen::MatrixXd bderiv(4, 18 + 3*nedgedofs);
    std::vector<Eigen::MatrixXd > bhess;
    Eigen::Matrix2d b = sff.secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL, hessian ? &bhess : NULL);
    
    Eigen::Matrix<double, 4, 9> aderivsmall;
    std::vector<Eigen::Matrix<double, 9, 9> > ahesssmall;
    Eigen::Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderivsmall : NULL, hessian ? &ahesssmall : NULL);
    Eigen::MatrixXd aderiv(4, 18 + 3 * nedgedofs);
    if (derivative || hessian)
    {
        aderiv.setZero();
        aderiv.block(0, 0, 4, 9) = aderivsmall;
    }
    std::vector<Eigen::MatrixXd> ahess = bhess;
    if (hessian)
    {
        for (int i = 0; i < 4; i++)
        {
            ahess[i].setZero();
            ahess[i].block(0, 0, 9, 9) = ahesssmall[i];
        }
    }

    double rtdeta = std::sqrt(a.determinant());
    Eigen::Matrix2d abaradj = adjugate(abar);
    Eigen::Matrix2d bbaradj = adjugate(bbar);
    Eigen::Matrix2d aadj = adjugate(a);
    double rtdetabar = std::sqrt(abar.determinant());

    double result = coeff1 * rtdetabar * 2.0 * ((aadj*b / a.determinant() - abaradj * bbar / abar.determinant())*(aadj*b / a.determinant() - abaradj * bbar / abar.determinant())).trace();
    result += coeff2 * rtdetabar * 2.0 * (aadj*b / a.determinant() - abaradj * bbar / abar.determinant()).trace() * (aadj*b / a.determinant() - abaradj * bbar / abar.determinant()).trace();

    Eigen::Matrix2d badj = adjugate(b);

    
    if (derivative)
    {
        derivative->setZero();
       
        double term1 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant();
        
        Eigen::Matrix2d m1 = aadj * b*aadj;
        *derivative += term1 * m1(0, 0) * bderiv.row(0);
        *derivative += term1 * m1(0, 1) * bderiv.row(1);
        *derivative += term1 * m1(1, 0) * bderiv.row(2);
        *derivative += term1 * m1(1, 1) * bderiv.row(3);
 
        Eigen::Matrix2d m2 = badj * a * badj;
        *derivative += term1 * m2(0, 0) * aderiv.row(0);
        *derivative += term1 * m2(0, 1) * aderiv.row(1);
        *derivative += term1 * m2(1, 0) * aderiv.row(2);
        *derivative += term1 * m2(1, 1) * aderiv.row(3);
        
       double term2 = coeff1 * rtdetabar * -4.0 / a.determinant() / a.determinant() / a.determinant() * (aadj*b*aadj*b).trace();
        *derivative += term2 * aadj(0, 0) * aderiv.row(0);
        *derivative += term2 * aadj(0, 1) * aderiv.row(1);
        *derivative += term2 * aadj(1, 0) * aderiv.row(2);
        *derivative += term2 * aadj(1, 1) * aderiv.row(3);
        
        double term3 = coeff1 * rtdetabar * -4.0 / a.determinant();
        Eigen::Matrix2d m3 = aadj * bbar * abaradj / abar.determinant();
        *derivative += term3 * m3(0, 0) * bderiv.row(0);
        *derivative += term3 * m3(0, 1) * bderiv.row(1);
        *derivative += term3 * m3(1, 0) * bderiv.row(2);
        *derivative += term3 * m3(1, 1) * bderiv.row(3);
        
        Eigen::Matrix2d m4 = badj * abar * bbaradj / abar.determinant();
        *derivative += term3 * m4(0, 0) * aderiv.row(0);
        *derivative += term3 * m4(0, 1) * aderiv.row(1);
        *derivative += term3 * m4(1, 0) * aderiv.row(2);
        *derivative += term3 * m4(1, 1) * aderiv.row(3);
        
        double term4 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant() * (aadj*b*abaradj*bbar).trace() / abar.determinant();
        *derivative += term4 * aadj(0, 0) * aderiv.row(0);
        *derivative += term4 * aadj(0, 1) * aderiv.row(1);
        *derivative += term4 * aadj(1, 0) * aderiv.row(2);
        *derivative += term4 * aadj(1, 1) * aderiv.row(3);
        
        // end term 1

        double term5 = coeff2 * rtdetabar * 4.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() * -1.0 / a.determinant();
        *derivative += term5 * badj(0, 0) * aderiv.row(0);
        *derivative += term5 * badj(0, 1) * aderiv.row(1);
        *derivative += term5 * badj(1, 0) * aderiv.row(2);
        *derivative += term5 * badj(1, 1) * aderiv.row(3);

        *derivative += term5 * aadj(0, 0) * bderiv.row(0);
        *derivative += term5 * aadj(0, 1) * bderiv.row(1);
        *derivative += term5 * aadj(1, 0) * bderiv.row(2);
        *derivative += term5 * aadj(1, 1) * bderiv.row(3);

        double term6 = coeff2 * rtdetabar * 4.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() * 1.0 / a.determinant() / a.determinant() * (aadj*b).trace();
        *derivative += term6 * aadj(0, 0) * aderiv.row(0);
        *derivative += term6 * aadj(0, 1) * aderiv.row(1);
        *derivative += term6 * aadj(1, 0) * aderiv.row(2);
        *derivative += term6 * aadj(1, 1) * aderiv.row(3);
        
    }
    
    if (hessian)
    {
        hessian->setZero();               
        Eigen::MatrixXd aadjda = aadj(0, 0) * aderiv.row(0);
        aadjda += aadj(0, 1) * aderiv.row(1);
        aadjda += aadj(1, 0) * aderiv.row(2);
        aadjda += aadj(1, 1) * aderiv.row(3);
        
        double term1 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant();
        Eigen::Matrix2d m1 = aadj * b*aadj;
        *hessian += term1 * m1(0, 0) * bhess[0];
        *hessian += term1 * m1(0, 1) * bhess[1];
        *hessian += term1 * m1(1, 0) * bhess[2];
        *hessian += term1 * m1(1, 1) * bhess[3];
        Eigen::Matrix2d m2 = aadj * b;
        *hessian += term1 * (m2(0, 0) * aderiv.row(3).transpose() + m2(0, 1) * -aderiv.row(2).transpose()) * bderiv.row(0);
        *hessian += term1 * (m2(0, 0) * -aderiv.row(1).transpose() + m2(0, 1) * aderiv.row(0).transpose()) * bderiv.row(1);
        *hessian += term1 * (m2(1, 0) * aderiv.row(3).transpose() + m2(1, 1) * -aderiv.row(2).transpose()) * bderiv.row(2);
        *hessian += term1 * (m2(1, 0) * -aderiv.row(1).transpose() + m2(1, 1) * aderiv.row(0).transpose()) * bderiv.row(3);
        
        *hessian += term1 * (aadj(0, 0) * bderiv.row(0).transpose() + aadj(0, 1) * bderiv.row(2).transpose()) * (aadj(0, 0) * bderiv.row(0) + aadj(0, 1) * bderiv.row(2));
        *hessian += term1 * (aadj(1, 0) * bderiv.row(0).transpose() + aadj(1, 1) * bderiv.row(2).transpose()) * (aadj(0, 0) * bderiv.row(1) + aadj(0, 1) * bderiv.row(3));
        *hessian += term1 * (aadj(0, 0) * bderiv.row(1).transpose() + aadj(0, 1) * bderiv.row(3).transpose()) * (aadj(1, 0) * bderiv.row(0) + aadj(1, 1) * bderiv.row(2));
        *hessian += term1 * (aadj(1, 0) * bderiv.row(1).transpose() + aadj(1, 1) * bderiv.row(3).transpose()) * (aadj(1, 0) * bderiv.row(1) + aadj(1, 1) * bderiv.row(3));

        Eigen::Matrix2d m3 = a * badj;
        *hessian += term1 * (m3(1, 0) * aderiv.row(1).transpose() + m3(1, 1) * aderiv.row(3).transpose()) * bderiv.row(0);
        *hessian += term1 * -(m3(0, 0) * aderiv.row(1).transpose() + m3(0, 1) * aderiv.row(3).transpose()) * bderiv.row(1);
        *hessian += term1 * -(m3(1, 0) * aderiv.row(0).transpose() + m3(1, 1) * aderiv.row(2).transpose()) * bderiv.row(2);
        *hessian += term1 * (m3(0, 0) * aderiv.row(0).transpose() + m3(0, 1) * aderiv.row(2).transpose()) * bderiv.row(3);

        double term2 = coeff1 * rtdetabar * -8.0 / a.determinant() / a.determinant() / a.determinant();
        Eigen::Matrix2d m4 = aadj * b*aadj;
        
        Eigen::MatrixXd m4db = m4(0, 0) * bderiv.row(0);
        m4db += m4(0, 1) * bderiv.row(1);
        m4db += m4(1, 0) * bderiv.row(2);
        m4db += m4(1, 1) * bderiv.row(3);
        *hessian += term2 * aadjda.transpose() * m4db;
        
        double term3 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant();
        Eigen::Matrix2d m5 = badj * a*badj;
        *hessian += term3 * m5(0, 0) * ahess[0];
        *hessian += term3 * m5(0, 1) * ahess[1];
        *hessian += term3 * m5(1, 0) * ahess[2];
        *hessian += term3 * m5(1, 1) * ahess[3];

        Eigen::Matrix2d m6 = badj * a;
        *hessian += term3 * (m6(0, 0) * bderiv.row(3).transpose() + m6(0, 1) * -bderiv.row(2).transpose()) * aderiv.row(0);
        *hessian += term3 * (m6(0, 0) * -bderiv.row(1).transpose() + m6(0, 1) * bderiv.row(0).transpose()) * aderiv.row(1);
        *hessian += term3 * (m6(1, 0) * bderiv.row(3).transpose() + m6(1, 1) * -bderiv.row(2).transpose()) * aderiv.row(2);
        *hessian += term3 * (m6(1, 0) * -bderiv.row(1).transpose() + m6(1, 1) * bderiv.row(0).transpose()) * aderiv.row(3);

        *hessian += term3 * (badj(0, 0) * aderiv.row(0).transpose() + badj(0, 1) * aderiv.row(2).transpose()) * (badj(0, 0) * aderiv.row(0) + badj(0, 1) * aderiv.row(2));
        *hessian += term3 * (badj(1, 0) * aderiv.row(0).transpose() + badj(1, 1) * aderiv.row(2).transpose()) * (badj(0, 0) * aderiv.row(1) + badj(0, 1) * aderiv.row(3));
        *hessian += term3 * (badj(0, 0) * aderiv.row(1).transpose() + badj(0, 1) * aderiv.row(3).transpose()) * (badj(1, 0) * aderiv.row(0) + badj(1, 1) * aderiv.row(2));
        *hessian += term3 * (badj(1, 0) * aderiv.row(1).transpose() + badj(1, 1) * aderiv.row(3).transpose()) * (badj(1, 0) * aderiv.row(1) + badj(1, 1) * aderiv.row(3));

        Eigen::Matrix2d m7 = b * aadj;
        *hessian += term3 * (m7(1, 0) * bderiv.row(1).transpose() + m7(1, 1) * bderiv.row(3).transpose()) * aderiv.row(0);
        *hessian += term3 * -(m7(0, 0) * bderiv.row(1).transpose() + m7(0, 1) * bderiv.row(3).transpose()) * aderiv.row(1);
        *hessian += term3 * -(m7(1, 0) * bderiv.row(0).transpose() + m7(1, 1) * bderiv.row(2).transpose()) * aderiv.row(2);
        *hessian += term3 * (m7(0, 0) * bderiv.row(0).transpose() + m7(0, 1) * bderiv.row(2).transpose()) * aderiv.row(3);

        double term4 = coeff1 * rtdetabar * -8.0 / a.determinant() / a.determinant() / a.determinant();
        Eigen::Matrix2d m8 = badj * a*badj;
        Eigen::MatrixXd m8da = m8(0, 0) * aderiv.row(0);
        m8da += m8(0, 1) * aderiv.row(1);
        m8da += m8(1, 0) * aderiv.row(2);
        m8da += m8(1, 1) * aderiv.row(3);
        *hessian += term4 * aadjda.transpose() * m8da;

        double term5 = coeff1 * rtdetabar * -4.0 / a.determinant() / a.determinant() / a.determinant() * (aadj*b*aadj*b).trace();
        *hessian += term5 * aadj(0, 0) * ahess[0];
        *hessian += term5 * aadj(0, 1) * ahess[1];
        *hessian += term5 * aadj(1, 0) * ahess[2];
        *hessian += term5 * aadj(1, 1) * ahess[3];

        *hessian += term5 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term5 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term5 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term5 * aderiv.row(0).transpose() * aderiv.row(3);

        double term6 = coeff1 * rtdetabar * -8.0 / a.determinant() / a.determinant() / a.determinant();
        Eigen::Matrix2d m9 = aadj * b*aadj;
        Eigen::MatrixXd m9db = m9(0, 0) * bderiv.row(0);
        m9db += m9(0, 1) * bderiv.row(1);
        m9db += m9(1, 0) * bderiv.row(2);
        m9db += m9(1, 1) * bderiv.row(3);
        *hessian += term6 * m9db.transpose() * aadjda;

        Eigen::Matrix2d m10 = badj * a*badj;
        Eigen::MatrixXd m10da = m10(0, 0) * aderiv.row(0);
        m10da += m10(0, 1) * aderiv.row(1);
        m10da += m10(1, 0) * aderiv.row(2);
        m10da += m10(1, 1) * aderiv.row(3);
        *hessian += term6 * m10da.transpose() * aadjda;

        double term7 = coeff1 * rtdetabar * 12.0 / a.determinant() / a.determinant() / a.determinant() / a.determinant() * (aadj*b*aadj*b).trace();
        *hessian += term7 * aadjda.transpose() * aadjda;
        
        double term8 = coeff1 * rtdetabar * -4.0 / a.determinant();
        Eigen::Matrix2d m11 = abar * bbaradj / abar.determinant();
        *hessian += term8 * (m11(1, 0) * aderiv.row(1).transpose() + m11(1, 1) * aderiv.row(3).transpose()) * bderiv.row(0);
        *hessian += term8 * -(m11(0, 0) * aderiv.row(1).transpose() + m11(0, 1) * aderiv.row(3).transpose()) * bderiv.row(1);
        *hessian += term8 * -(m11(1, 0) * aderiv.row(0).transpose() + m11(1, 1) * aderiv.row(2).transpose()) * bderiv.row(2);
        *hessian += term8 * (m11(0, 0) * aderiv.row(0).transpose() + m11(0, 1) * aderiv.row(2).transpose()) * bderiv.row(3);

        Eigen::Matrix2d m12 = aadj * bbar*abaradj / abar.determinant();
        *hessian += term8 * m12(0, 0) * bhess[0];
        *hessian += term8 * m12(0, 1) * bhess[1];
        *hessian += term8 * m12(1, 0) * bhess[2];
        *hessian += term8 * m12(1, 1) * bhess[3];

        double term9 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant();
        Eigen::Matrix2d m13 = aadj * bbar * abaradj / abar.determinant();
        Eigen::MatrixXd m13db = m13(0, 0) * bderiv.row(0);
        m13db += m13(0, 1) * bderiv.row(1);
        m13db += m13(1, 0) * bderiv.row(2);
        m13db += m13(1, 1) * bderiv.row(3);

        *hessian += term9 * aadjda.transpose() * m13db;

        double term10 = coeff1 * rtdetabar * -4.0 / a.determinant();
        Eigen::Matrix2d m14 = bbar * abaradj / abar.determinant();
        *hessian += term10 * (m14(1, 0) * bderiv.row(1).transpose() + m14(1, 1) * bderiv.row(3).transpose()) * aderiv.row(0);
        *hessian += term10 * -(m14(0, 0) * bderiv.row(1).transpose() + m14(0, 1) * bderiv.row(3).transpose()) * aderiv.row(1);
        *hessian += term10 * -(m14(1, 0) * bderiv.row(0).transpose() + m14(1, 1) * bderiv.row(2).transpose()) * aderiv.row(2);
        *hessian += term10 * (m14(0, 0) * bderiv.row(0).transpose() + m14(0, 1) * bderiv.row(2).transpose()) * aderiv.row(3);

        Eigen::Matrix2d m15 = badj * abar * bbaradj / abar.determinant();
        *hessian += term10 * m15(0, 0) * ahess[0];
        *hessian += term10 * m15(0, 1) * ahess[1];
        *hessian += term10 * m15(1, 0) * ahess[2];
        *hessian += term10 * m15(1, 1) * ahess[3];

        double term11 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant();
        Eigen::Matrix2d m16 = badj * abar * bbaradj / abar.determinant();
        Eigen::MatrixXd m16da = m16(0, 0) * aderiv.row(0);
        m16da += m16(0, 1) * aderiv.row(1);
        m16da += m16(1, 0) * aderiv.row(2);
        m16da += m16(1, 1) * aderiv.row(3);

        *hessian += term11 * aadjda.transpose() * m16da;
        
        *hessian += term11 * m16da.transpose() * aadjda;
        *hessian += term11 * m13db.transpose() * aadjda;

        double term12 = coeff1 * rtdetabar * 4.0 / a.determinant() / a.determinant() * (aadj*b*abaradj*bbar).trace() / abar.determinant();
        *hessian += term12 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term12 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term12 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term12 * aderiv.row(0).transpose() * aderiv.row(3);
        *hessian += term12 * aadj(0, 0) * ahess[0];
        *hessian += term12 * aadj(0, 1) * ahess[1];
        *hessian += term12 * aadj(1, 0) * ahess[2];
        *hessian += term12 * aadj(1, 1) * ahess[3];

        double term13 = coeff1 * rtdetabar * -8.0 / a.determinant() / a.determinant() / a.determinant() * (aadj*b*abaradj*bbar).trace() / abar.determinant();
        *hessian += term13 * aadjda.transpose() * aadjda;
        
        // end term 1

        double term14 = coeff2 * rtdetabar * -4.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() / a.determinant();
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

        double term15 = coeff2 * rtdetabar * 4.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() / a.determinant() / a.determinant();
        Eigen::MatrixXd badjda = badj(0, 0) * aderiv.row(0);
        badjda += badj(0, 1) * aderiv.row(1);
        badjda += badj(1, 0) * aderiv.row(2);
        badjda += badj(1, 1) * aderiv.row(3);
        *hessian += term15 * aadjda.transpose() * badjda;
        Eigen::MatrixXd aadjdb = aadj(0, 0) * bderiv.row(0);
        aadjdb += aadj(0, 1) * bderiv.row(1);
        aadjdb += aadj(1, 0) * bderiv.row(2);
        aadjdb += aadj(1, 1) * bderiv.row(3);
        *hessian += term15 * aadjda.transpose() * aadjdb;

        double term16 = coeff2 * rtdetabar * 4.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() / a.determinant() / a.determinant() * (aadj*b).trace();
        *hessian += term16 * aadj(0, 0) * ahess[0];
        *hessian += term16 * aadj(1, 0) * ahess[1];
        *hessian += term16 * aadj(0, 1) * ahess[2];
        *hessian += term16 * aadj(1, 1) * ahess[3];
        *hessian += term16 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term16 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term16 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term16 * aderiv.row(0).transpose() * aderiv.row(3);

        double term17 = coeff2 * rtdetabar * 4.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() / a.determinant() / a.determinant();
        *hessian += term17 * aadjdb.transpose() * aadjda;
        *hessian += term17 * badjda.transpose() * aadjda;

        double term18 = coeff2 * rtdetabar * -8.0 * (abaradj*bbar / abar.determinant() - aadj * b / a.determinant()).trace() / a.determinant() / a.determinant() / a.determinant() * (aadj*b).trace();
        *hessian += term18 * aadjda.transpose() * aadjda;

        double term19 = coeff2 * rtdetabar * 4.0 / a.determinant() / a.determinant();
        *hessian += term19 * badjda.transpose() * badjda;
        *hessian += term19 * badjda.transpose() * aadjdb;
        *hessian += term19 * aadjdb.transpose() * badjda;
        *hessian += term19 * aadjdb.transpose() * aadjdb;

        double term20 = coeff2 * rtdetabar * -4.0 / a.determinant() / a.determinant() / a.determinant() * (aadj*b).trace();
        *hessian += term20 * aadjda.transpose() * badjda;
        *hessian += term20 * aadjda.transpose() * aadjdb;
        *hessian += term20 * badjda.transpose() * aadjda;
        *hessian += term20 * aadjdb.transpose() * aadjda;

        double term21 = coeff2 * rtdetabar * 4.0 / a.determinant() / a.determinant() / a.determinant() / a.determinant() * (aadj*b).trace() * (aadj*b).trace();
        *hessian += term21 * aadjda.transpose() * aadjda;
    }   

    return result;
}

