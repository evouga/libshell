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
    double result = coeff1 * (((abaradj*(a - abar)).trace() + 2 * abar.determinant()) / rtdeta - 2.0 * rtdetabar);

    double coeff2 = thickness * lameBeta_ / 2.0;
    result += coeff2 * (rtdetabar - rtdeta)*(rtdetabar - rtdeta) / rtdetabar;

    if (derivative)
    {
        derivative->setZero();
        *derivative += coeff1 / rtdeta * abaradj(0,0) * aderiv.row(0).transpose();
        *derivative += coeff1 / rtdeta * abaradj(1,0) * aderiv.row(1).transpose();
        *derivative += coeff1 / rtdeta * abaradj(0,1) * aderiv.row(2).transpose();
        *derivative += coeff1 / rtdeta * abaradj(1,1) * aderiv.row(3).transpose();

        double term2 = -coeff1 / 2.0 / rtdeta / rtdeta / rtdeta * ((abaradj*(a - abar)).trace() + 2 * abar.determinant());
        Eigen::Matrix2d aadj = adjugate(a);
        *derivative += term2 * aadj(0, 0) * aderiv.row(0).transpose();
        *derivative += term2 * aadj(1, 0) * aderiv.row(1).transpose();
        *derivative += term2 * aadj(0, 1) * aderiv.row(2).transpose();
        *derivative += term2 * aadj(1, 1) * aderiv.row(3).transpose();      

        double term3 = coeff2 / rtdetabar / rtdeta * -1.0 * (rtdetabar - rtdeta);
        *derivative += term3 * aadj(0, 0) * aderiv.row(0).transpose();
        *derivative += term3 * aadj(1, 0) * aderiv.row(1).transpose();
        *derivative += term3 * aadj(0, 1) * aderiv.row(2).transpose();
        *derivative += term3 * aadj(1, 1) * aderiv.row(3).transpose();
    }

    if (hessian)
    {
        hessian->setZero();
        Eigen::Matrix<double, 1, 9> innerbar = abaradj(0, 0) * aderiv.row(0).transpose();
        innerbar += abaradj(1, 0) * aderiv.row(1).transpose();
        innerbar += abaradj(0, 1) * aderiv.row(2).transpose();
        innerbar += abaradj(1, 1) * aderiv.row(3).transpose();

        Eigen::Matrix2d aadj = adjugate(a);
        Eigen::Matrix<double, 1, 9> inner = aadj(0, 0) * aderiv.row(0).transpose();
        inner += aadj(1, 0) * aderiv.row(1).transpose();
        inner += aadj(0, 1) * aderiv.row(2).transpose();
        inner += aadj(1, 1) * aderiv.row(3).transpose();

        *hessian += coeff1 / rtdeta * abaradj(0, 0) * ahess[0];
        *hessian += coeff1 / rtdeta * abaradj(1, 0) * ahess[1];
        *hessian += coeff1 / rtdeta * abaradj(0, 1) * ahess[2];
        *hessian += coeff1 / rtdeta * abaradj(1, 1) * ahess[3];

        *hessian += coeff1 * -1.0 / 2.0 / rtdeta / rtdeta / rtdeta * inner.transpose() * innerbar;
        *hessian += coeff1 * -1.0 / 2.0 / rtdeta / rtdeta / rtdeta * innerbar.transpose() * inner;
        *hessian += coeff1 * 3.0 / 4.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * ((abaradj*(a - abar)).trace() + 2 * abar.determinant()) * inner.transpose() * inner;

        double term2 = -coeff1 / 2.0 / rtdeta / rtdeta / rtdeta * ((abaradj*(a - abar)).trace() + 2 * abar.determinant());
        *hessian += term2 * aadj(0, 0) * ahess[0];
        *hessian += term2 * aadj(1, 0) * ahess[1];
        *hessian += term2 * aadj(0, 1) * ahess[2];
        *hessian += term2 * aadj(1, 1) * ahess[3];

        *hessian += term2 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term2 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term2 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term2 * aderiv.row(0).transpose() * aderiv.row(3);

        *hessian += coeff2 / 2.0 / rtdeta / rtdeta / rtdeta * inner.transpose() * inner;

        double term3 = -coeff2 * (rtdetabar - rtdeta) / rtdetabar / rtdeta;

        *hessian += term3 * aadj(0, 0) * ahess[0];
        *hessian += term3 * aadj(1, 0) * ahess[1];
        *hessian += term3 * aadj(0, 1) * ahess[2];
        *hessian += term3 * aadj(1, 1) * ahess[3];

        *hessian += term3 * aderiv.row(3).transpose() * aderiv.row(0);
        *hessian += term3 * -aderiv.row(1).transpose() * aderiv.row(1);
        *hessian += term3 * -aderiv.row(2).transpose() * aderiv.row(2);
        *hessian += term3 * aderiv.row(0).transpose() * aderiv.row(3);

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

    double result = coeff1 * (
        (4.0 * (bbaradj * (b - bbar)).trace() + 8.0 * bbar.determinant()) / rtdeta  //1A
        - (2.0 * (abaradj * (b + bbar)).trace() * (aadj*b).trace() + 4.0 * abar.determinant() * b.determinant()) / rtdeta / rtdeta / rtdeta //1B
        + 3.0 * (aadj*b).trace() * (aadj*b).trace() * abar.determinant() / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta
        + (abaradj*bbar).trace() * (abaradj*bbar).trace() / rtdetabar / rtdetabar / rtdetabar
        - 4.0 * bbar.determinant() / rtdetabar
        );

    
if (derivative)
    {
        derivative->setZero();
       
        *derivative += coeff1 * 4.0 / rtdeta * bbaradj(0, 0) * bderiv.row(0);
        *derivative += coeff1 * 4.0 / rtdeta * bbaradj(1, 0) * bderiv.row(1);
        *derivative += coeff1 * 4.0 / rtdeta * bbaradj(0, 1) * bderiv.row(2);
        *derivative += coeff1 * 4.0 / rtdeta * bbaradj(1, 1) * bderiv.row(3);
        
        double term2 = -2.0 * coeff1 / rtdeta / rtdeta / rtdeta * ((bbaradj * (b - bbar)).trace() + 2.0 * bbar.determinant());
        *derivative += term2 * aadj(0, 0) * aderiv.row(0);
        *derivative += term2 * aadj(1, 0) * aderiv.row(1);
        *derivative += term2 * aadj(0, 1) * aderiv.row(2);
        *derivative += term2 * aadj(1, 1) * aderiv.row(3);
           
        //// end 1A

        
        
        double term3 = -2.0 * coeff1 / rtdeta / rtdeta / rtdeta * (aadj*b).trace();
        
        *derivative += term3 * abaradj(0, 0) * bderiv.row(0);
        *derivative += term3 * abaradj(1, 0) * bderiv.row(1);
        *derivative += term3 * abaradj(0, 1) * bderiv.row(2);
        *derivative += term3 * abaradj(1, 1) * bderiv.row(3);
        
        
        double term4 = -2.0 * coeff1 / rtdeta / rtdeta / rtdeta * (abaradj*(b + bbar)).trace(); // + sign is not a bug
        Eigen::Matrix2d badj = adjugate(b);
        
        *derivative += term4 * aadj(0, 0) * bderiv.row(0);
        *derivative += term4 * aadj(1, 0) * bderiv.row(1);
        *derivative += term4 * aadj(0, 1) * bderiv.row(2);
        *derivative += term4 * aadj(1, 1) * bderiv.row(3);
        *derivative += term4 * badj(0, 0) * aderiv.row(0);
        *derivative += term4 * badj(1, 0) * aderiv.row(1);
        *derivative += term4 * badj(0, 1) * aderiv.row(2);
        *derivative += term4 * badj(1, 1) * aderiv.row(3);

        double term5 = -4.0 * coeff1 / rtdeta / rtdeta / rtdeta * abar.determinant();
        *derivative += term5 * badj(0, 0) * bderiv.row(0);
        *derivative += term5 * badj(1, 0) * bderiv.row(1);
        *derivative += term5 * badj(0, 1) * bderiv.row(2);
        *derivative += term5 * badj(1, 1) * bderiv.row(3);
        
        double term6 = 3.0 * coeff1 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * ((abaradj*(b + bbar)).trace() * (aadj*b).trace() + 2.0 * abar.determinant() * b.determinant());
        *derivative += term6 * aadj(0, 0) * aderiv.row(0);
        *derivative += term6 * aadj(1, 0) * aderiv.row(1);
        *derivative += term6 * aadj(0, 1) * aderiv.row(2);
        *derivative += term6 * aadj(1, 1) * aderiv.row(3);
        
        //// end 1B
        
        double term7 = 6.0 * coeff1 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * abar.determinant() * (aadj*b).trace();
        *derivative += term7 * aadj(0, 0) * bderiv.row(0);
        *derivative += term7 * aadj(1, 0) * bderiv.row(1);
        *derivative += term7 * aadj(0, 1) * bderiv.row(2);
        *derivative += term7 * aadj(1, 1) * bderiv.row(3);
        *derivative += term7 * badj(0, 0) * aderiv.row(0);
        *derivative += term7 * badj(1, 0) * aderiv.row(1);
        *derivative += term7 * badj(0, 1) * aderiv.row(2);
        *derivative += term7 * badj(1, 1) * aderiv.row(3);

        double term8 = -15.0 * coeff1 / 2.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * (aadj*b).trace() * (aadj*b).trace() * abar.determinant();
        *derivative += term8 * aadj(0, 0) * aderiv.row(0);
        *derivative += term8 * aadj(1, 0) * aderiv.row(1);
        *derivative += term8 * aadj(0, 1) * aderiv.row(2);
        *derivative += term8 * aadj(1, 1) * aderiv.row(3);
        
    }
    
    if (hessian)
    {
        hessian->setZero();
        
        *hessian += coeff1 * 4.0 / rtdeta * bbaradj(0, 0) * bhess[0];
        *hessian += coeff1 * 4.0 / rtdeta * bbaradj(1, 0) * bhess[1];
        *hessian += coeff1 * 4.0 / rtdeta * bbaradj(0, 1) * bhess[2];
        *hessian += coeff1 * 4.0 / rtdeta * bbaradj(1, 1) * bhess[3];
        
        Eigen::MatrixXd aadjda = aadj(0, 0)*aderiv.row(0);
        aadjda += aadj(1, 0)*aderiv.row(1);
        aadjda += aadj(0, 1)*aderiv.row(2);
        aadjda += aadj(1, 1)*aderiv.row(3);
        
        Eigen::MatrixXd bbaradjdb = bbaradj(0, 0)*bderiv.row(0);
        bbaradjdb += bbaradj(1, 0)*bderiv.row(1);
        bbaradjdb += bbaradj(0, 1)*bderiv.row(2);
        bbaradjdb += bbaradj(1, 1)*bderiv.row(3);
        
        double term2 = coeff1 * -2.0 / rtdeta / rtdeta / rtdeta;
        *hessian += term2 * aadjda.transpose() * bbaradjdb;
        *hessian += term2 * bbaradjdb.transpose() * aadjda;

        
        double term3 = coeff1 * 3.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * ((bbaradj*(b - bbar)).trace() + 2.0 * bbar.determinant());
        *hessian += term3 * aadjda.transpose() * aadjda;
        
        double term4 = coeff1 * -2.0 / rtdeta / rtdeta / rtdeta * ((bbaradj*(b - bbar)).trace() + 2.0 * bbar.determinant());
        
        
        *hessian += term4 * aadj(0, 0) * ahess[0];
        *hessian += term4 * aadj(1, 0) * ahess[1];
        *hessian += term4 * aadj(0, 1) * ahess[2];
        *hessian += term4 * aadj(1, 1) * ahess[3];
        

        Eigen::MatrixXd daadjda = aderiv.row(3).transpose() * aderiv.row(0);
        daadjda -= aderiv.row(1).transpose() * aderiv.row(1);
        daadjda -= aderiv.row(2).transpose() * aderiv.row(2);
        daadjda += aderiv.row(0).transpose() * aderiv.row(3);
        
        
        *hessian += term4 * daadjda;

        //// end 1A
        
        double term5 = coeff1 * -2.0 / rtdeta / rtdeta / rtdeta;
        *hessian += term5 * (aadj*b).trace() * abaradj(0, 0) * bhess[0];
        *hessian += term5 * (aadj*b).trace() * abaradj(1, 0) * bhess[1];
        *hessian += term5 * (aadj*b).trace() * abaradj(0, 1) * bhess[2];
        *hessian += term5 * (aadj*b).trace() * abaradj(1, 1) * bhess[3];
        
        Eigen::MatrixXd aadjdb = aadj(0, 0) * bderiv.row(0);
        aadjdb += aadj(1, 0) * bderiv.row(1);
        aadjdb += aadj(0, 1) * bderiv.row(2);
        aadjdb += aadj(1, 1) * bderiv.row(3);

        Eigen::MatrixXd abaradjdb = abaradj(0, 0) * bderiv.row(0);
        abaradjdb += abaradj(1, 0) * bderiv.row(1);
        abaradjdb += abaradj(0, 1) * bderiv.row(2);
        abaradjdb += abaradj(1, 1) * bderiv.row(3);

        Eigen::Matrix2d badj = adjugate(b);
        Eigen::MatrixXd badjda = badj(0, 0) * aderiv.row(0);
        badjda += badj(1, 0) * aderiv.row(1);
        badjda += badj(0, 1) * aderiv.row(2);
        badjda += badj(1, 1) * aderiv.row(3);

        *hessian += term5 * aadjdb.transpose() * abaradjdb;
        *hessian += term5 * badjda.transpose() * abaradjdb;
        *hessian += term5 * abaradjdb.transpose() * aadjdb;
        *hessian += term5 * abaradjdb.transpose() * badjda;
        *hessian += term5 * (abaradj * (b + bbar)).trace() * aadj(0, 0) * bhess[0];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * aadj(1, 0) * bhess[1];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * aadj(0, 1) * bhess[2];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * aadj(1, 1) * bhess[3];
  
        Eigen::MatrixXd daadjdb = aderiv.row(3).transpose() * bderiv.row(0);
        daadjdb -= aderiv.row(1).transpose() * bderiv.row(1);
        daadjdb -= aderiv.row(2).transpose() * bderiv.row(2);
        daadjdb += aderiv.row(0).transpose() * bderiv.row(3);
    
        *hessian += term5 * (abaradj * (b + bbar)).trace() * daadjdb;
        *hessian += term5 * (abaradj * (b + bbar)).trace() * badj(0, 0) * ahess[0];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * badj(1, 0) * ahess[1];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * badj(0, 1) * ahess[2];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * badj(1, 1) * ahess[3];
        *hessian += term5 * (abaradj * (b + bbar)).trace() * daadjdb.transpose();

        *hessian += term5 * 2.0 * abar.determinant() * badj(0, 0) * bhess[0];
        *hessian += term5 * 2.0 * abar.determinant() * badj(1, 0) * bhess[1];
        *hessian += term5 * 2.0 * abar.determinant() * badj(0, 1) * bhess[2];
        *hessian += term5 * 2.0 * abar.determinant() * badj(1, 1) * bhess[3];
        
        Eigen::MatrixXd dbadjdb = bderiv.row(3).transpose() * bderiv.row(0);
        dbadjdb -= bderiv.row(1).transpose() * bderiv.row(1);
        dbadjdb -= bderiv.row(2).transpose() * bderiv.row(2);
        dbadjdb += bderiv.row(0).transpose() * bderiv.row(3);

        *hessian += term5 * 2.0 * abar.determinant() * dbadjdb;

        double term6 = coeff1 * 3.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta;
        
        *hessian += term6 * (aadj*b).trace() * aadjda.transpose() * abaradjdb;
        *hessian += term6 * (abaradj*(b+bbar)).trace() * aadjda.transpose() * aadjdb;
        *hessian += term6 * (abaradj*(b+bbar)).trace() * aadjda.transpose() * badjda;
        

        Eigen::MatrixXd badjdb = badj(0, 0) * bderiv.row(0);
        badjdb += badj(1, 0) * bderiv.row(1);
        badjdb += badj(0, 1) * bderiv.row(2);
        badjdb += badj(1, 1) * bderiv.row(3);

        
        *hessian += term6 * 2.0 * abar.determinant() * aadjda.transpose() * badjdb;
        
        *hessian += term6 * (aadj*b).trace() * abaradjdb.transpose() * aadjda;
        *hessian += term6 * (abaradj*(b + bbar)).trace() * aadjdb.transpose() * aadjda;
        *hessian += term6 * (abaradj*(b + bbar)).trace() * badjda.transpose() * aadjda;
        *hessian += term6 * 2.0 * abar.determinant() * badjdb.transpose() * aadjda;

        double term6b = term6 * ((abaradj*(b + bbar)).trace()*(aadj*b).trace() + 2.0 * abar.determinant() * b.determinant());
        *hessian += term6b * aadj(0, 0) * ahess[0];
        *hessian += term6b * aadj(1, 0) * ahess[1];
        *hessian += term6b * aadj(0, 1) * ahess[2];
        *hessian += term6b * aadj(1, 1) * ahess[3];

        *hessian += term6b * daadjda;
        
        double term7 = coeff1 * -15.0 / 2.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * ((abaradj*(b + bbar)).trace() * (aadj*b).trace() + 2.0*abar.determinant() * b.determinant());
        *hessian += term7 * aadjda.transpose() * aadjda;
        
        //// end 1B
        
        double term8 = coeff1 * 6.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * abar.determinant();
        *hessian += term8 * aadjdb.transpose() * aadjdb;
        *hessian += term8 * aadjdb.transpose() * badjda;
        *hessian += term8 * badjda.transpose() * aadjdb;
        *hessian += term8 * badjda.transpose() * badjda;
        *hessian += term8 * (aadj*b).trace() * aadj(0, 0) * bhess[0];
        *hessian += term8 * (aadj*b).trace() * aadj(1, 0) * bhess[1];
        *hessian += term8 * (aadj*b).trace() * aadj(0, 1) * bhess[2];
        *hessian += term8 * (aadj*b).trace() * aadj(1, 1) * bhess[3];
        *hessian += term8 * (aadj*b).trace() * daadjdb;
        *hessian += term8 * (aadj*b).trace() * daadjdb.transpose();
        *hessian += term8 * (aadj*b).trace() * badj(0, 0) * ahess[0];
        *hessian += term8 * (aadj*b).trace() * badj(1, 0) * ahess[1];
        *hessian += term8 * (aadj*b).trace() * badj(0, 1) * ahess[2];
        *hessian += term8 * (aadj*b).trace() * badj(1, 1) * ahess[3];
        
        double term9 = coeff1 * -15.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * abar.determinant() * (aadj*b).trace();
        *hessian += term9 * aadjda.transpose() * aadjdb;
        *hessian += term9 * aadjda.transpose() * badjda;
        *hessian += term9 * aadjdb.transpose() * aadjda;
        *hessian += term9 * badjda.transpose() * aadjda;
        *hessian += term9 * 0.5 * (aadj*b).trace() * aadj(0, 0) * ahess[0];
        *hessian += term9 * 0.5 * (aadj*b).trace() * aadj(1, 0) * ahess[1];
        *hessian += term9 * 0.5 * (aadj*b).trace() * aadj(0, 1) * ahess[2];
        *hessian += term9 * 0.5 * (aadj*b).trace() * aadj(1, 1) * ahess[3];
        *hessian += term9 * 0.5 * (aadj*b).trace() * daadjda;

        double term10 = coeff1 * 105.0 / 4.0 / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta / rtdeta * (aadj*b).trace() * (aadj*b).trace() * abar.determinant();
        *hessian += term10 * aadjda.transpose() * aadjda;
        
    }    
    return result;
}

