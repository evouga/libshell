#include "../../include/StVKMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include "../../include/SecondFundamentalFormDiscretization.h"

double StVKMaterial::stretchingEnergy(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,    
    double thickness,
    const Eigen::Matrix2d &abar,
    int face,
    Eigen::Matrix<double, 1, 9> *derivative, // F(face, i)
    Eigen::Matrix<double, 9, 9> *hessian) const
{
    double coeff = thickness / 4.0;
    Eigen::Matrix2d abarinv = abar.inverse();
    Eigen::Matrix<double, 4, 9> aderiv;
    std::vector<Eigen::Matrix<double, 9, 9> > ahess;
    Eigen::Matrix2d a = firstFundamentalForm(mesh, curPos, face, (derivative || hessian) ? &aderiv : NULL, hessian ? &ahess : NULL);
    Eigen::Matrix2d M = abarinv * (a - abar);
    double dA = 0.5 * sqrt(abar.determinant());

    double StVK = 0.5 * lameAlpha_ * M.trace() * M.trace() + lameBeta_ * (M*M).trace();
    double result = coeff * dA * StVK;

    if (derivative)
    {
        derivative->setZero();
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(0,0) * aderiv.row(0).transpose();
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(1,0) * aderiv.row(1).transpose();
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(0,1) * aderiv.row(2).transpose();
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(1,1) * aderiv.row(3).transpose();
        Eigen::Matrix2d Mainv = M*abarinv;
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(0, 0) * aderiv.row(0).transpose();
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(1, 0) * aderiv.row(1).transpose();
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(0, 1) * aderiv.row(2).transpose();
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(1, 1) * aderiv.row(3).transpose();
    }

    if (hessian)
    {
        hessian->setZero();
        Eigen::Matrix<double, 1, 9> inner = abarinv(0,0) * aderiv.row(0).transpose();
        inner += abarinv(1,0) * aderiv.row(1).transpose();
        inner += abarinv(0,1) * aderiv.row(2).transpose();
        inner += abarinv(1,1) * aderiv.row(3).transpose();
        *hessian += coeff*dA * lameAlpha_ * inner.transpose() * inner;
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(0,0) * ahess[0];
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(1,0) * ahess[1];
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(0,1) * ahess[2];
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(1,1) * ahess[3];        
        Eigen::Matrix<double, 1, 9> inner00 = abarinv(0, 0) * aderiv.row(0) + abarinv(0, 1) * aderiv.row(2);        
        Eigen::Matrix<double, 1, 9> inner01 = abarinv(0, 0) * aderiv.row(1) + abarinv(0, 1) * aderiv.row(3);
        Eigen::Matrix<double, 1, 9> inner10 = abarinv(1, 0) * aderiv.row(0) + abarinv(1, 1) * aderiv.row(2);
        Eigen::Matrix<double, 1, 9> inner11 = abarinv(1, 0) * aderiv.row(1) + abarinv(1, 1) * aderiv.row(3);
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner00.transpose() * inner00;
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner01.transpose() * inner10;
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner10.transpose() * inner01;
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner11.transpose() * inner11;
        Eigen::Matrix2d Mainv = M*abarinv;
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(0, 0) * ahess[0];
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(1, 0) * ahess[1];
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(0, 1) * ahess[2];
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(1, 1) * ahess[3];
    }

    return result;
}

double StVKMaterial::bendingEnergy(
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
    double coeff = thickness*thickness*thickness / 12.0;
    int nedgedofs = sff.numExtraDOFs();
    Eigen::Matrix2d abarinv = abar.inverse();
    Eigen::MatrixXd bderiv(4, 18 + 3*nedgedofs);
    std::vector<Eigen::MatrixXd > bhess;
    Eigen::Matrix2d b = sff.secondFundamentalForm(mesh, curPos, extraDOFs, face, (derivative || hessian) ? &bderiv : NULL, hessian ? &bhess : NULL);
    Eigen::Matrix2d M = abarinv * (b - bbar);
    double dA = 0.5 * sqrt(abar.determinant());

    double StVK = 0.5 * lameAlpha_ * M.trace() * M.trace() + lameBeta_ * (M*M).trace();
    double result = coeff * dA * StVK;

    if (derivative)
    {
        derivative->setZero();
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(0,0) * bderiv.row(0);
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(1,0) * bderiv.row(1);
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(0,1) * bderiv.row(2);
        *derivative += coeff*dA * lameAlpha_ * M.trace() * abarinv(1,1) * bderiv.row(3);
        Eigen::Matrix2d Mainv = M*abarinv;
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(0, 0) * bderiv.row(0);
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(1, 0) * bderiv.row(1);
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(0, 1) * bderiv.row(2);
        *derivative += coeff*dA* 2.0 * lameBeta_ * Mainv(1, 1) * bderiv.row(3);
    }

    if (hessian)
    {
        hessian->setZero();
        Eigen::MatrixXd inner = abarinv(0,0) * bderiv.row(0);
        inner += abarinv(1,0) * bderiv.row(1);
        inner += abarinv(0,1) * bderiv.row(2);
        inner += abarinv(1,1) * bderiv.row(3);
        *hessian += coeff*dA * lameAlpha_ * inner.transpose() * inner;
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(0,0) * bhess[0];
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(1,0) * bhess[1];
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(0,1) * bhess[2];
        *hessian += coeff * dA * lameAlpha_ * M.trace() * abarinv(1,1) * bhess[3];        
        Eigen::MatrixXd inner00 = abarinv(0, 0) * bderiv.row(0) + abarinv(0, 1) * bderiv.row(2);        
        Eigen::MatrixXd inner01 = abarinv(0, 0) * bderiv.row(1) + abarinv(0, 1) * bderiv.row(3);
        Eigen::MatrixXd inner10 = abarinv(1, 0) * bderiv.row(0) + abarinv(1, 1) * bderiv.row(2);
        Eigen::MatrixXd inner11 = abarinv(1, 0) * bderiv.row(1) + abarinv(1, 1) * bderiv.row(3);
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner00.transpose() * inner00;
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner01.transpose() * inner10;
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner10.transpose() * inner01;
        *hessian += coeff * dA * 2.0 * lameBeta_ * inner11.transpose() * inner11;
        Eigen::Matrix2d Mainv = M*abarinv;
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(0, 0) * bhess[0];
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(1, 0) * bhess[1];
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(0, 1) * bhess[2];
        *hessian += coeff * dA * 2.0 * lameBeta_ * Mainv(1, 1) * bhess[3];
    }

    return result;
}

