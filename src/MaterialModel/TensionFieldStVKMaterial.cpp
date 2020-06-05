#include "../../include/TensionFieldStVKMaterial.h"
#include "../../include/MeshConnectivity.h"
#include <vector>
#include "../GeometryDerivatives.h"
#include <Eigen/Dense>
#include "../../include/MidedgeAngleSinFormulation.h"
#include "../../include/MidedgeAngleTanFormulation.h"
#include "../../include/MidedgeAverageFormulation.h"

template <class SFF>
double TensionFieldStVKMaterial<SFF>::stretchingEnergy(
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

    double T = M.trace();
    double D = M.determinant();
    double detAbarinv = abarinv.determinant();

    double lambda1 = T / 2.0 + sqrt(std::max(0.0, T*T / 4.0 - D));
    double lambda2 = T / 2.0 - sqrt(std::max(0.0, T*T / 4.0 - D));
    double sign = 1.0;

    //make lambda1 the largest eigen value
    if (lambda2 > lambda1)
    {
        std::swap(lambda1, lambda2); 
        sign = -1.0;
    }

    bool puretension = false;    

    double kstretch1 = 0.5 * coeff * lameAlpha_;
    double kstretch2 = coeff * lameBeta_;

    double transitionCoeff = - kstretch1 / (kstretch1 + kstretch2);

    if (lambda1 >= 0 && lambda2 >= transitionCoeff * lambda1)
    {
        puretension = true;    
    }

    if (puretension)
    {

        double StVK = 0.5 * lameAlpha_ * pow(M.trace(), 2) + lameBeta_ * (M * M).trace();
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

            Matrix2d Mainv = M * abarinv;
            for (int i = 0; i < 4; ++i) // iterate over Mainv and abarinv as if they were vectors
                *hessian += (lameAlpha_ * M.trace() * abarinv(i) + 2 * lameBeta_ * Mainv(i)) * ahess[i];

            Matrix<double, 1, 9> inner00 = abarinv(0, 0) * aderiv.row(0) + abarinv(0, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner01 = abarinv(0, 0) * aderiv.row(1) + abarinv(0, 1) * aderiv.row(3);
            Matrix<double, 1, 9> inner10 = abarinv(1, 0) * aderiv.row(0) + abarinv(1, 1) * aderiv.row(2);
            Matrix<double, 1, 9> inner11 = abarinv(1, 0) * aderiv.row(1) + abarinv(1, 1) * aderiv.row(3);
            *hessian += 2 * lameBeta_ * inner00.transpose() * inner00;
            *hessian += 2 * lameBeta_ * (inner01.transpose() * inner10 + inner10.transpose() * inner01);
            *hessian += 2 * lameBeta_ * inner11.transpose() * inner11;

            *hessian *= coeff * dA;
        }

        return result;
    }
    else if (lambda1 < 0)
    {
        if (derivative)
            derivative->setZero();
        if (hessian)
            hessian->setZero();
        return 0.0;
    }
    else
    {
        double lambda = lambda1;
        double kstretching = kstretch1 + kstretch2 - kstretch1 * kstretch1 / (kstretch1 + kstretch2);

        double result = kstretching * 0.5 * dA * lambda * lambda;
        if (derivative || hessian)
        {
            double denom = sqrt(T * T / 4.0 - D);
            Eigen::Matrix2d adjstrain;
            adjstrain(0, 0) = (a - abar)(1, 1);
            adjstrain(1, 1) = (a - abar)(0, 0);
            adjstrain(0, 1) = -(a - abar)(1, 0);
            adjstrain(1, 0) = -(a - abar)(0, 1);
            Eigen::Matrix2d mat = 0.5 * abarinv + sign / denom * (T / 4.0 * abarinv - 1.0 / 2.0 * detAbarinv * adjstrain); // lamda equals (tr(A) + sqrt(tr(A)^2 - 4 * det(A)))/2

            if (derivative)
            {
                derivative->setZero();

                (*derivative) += kstretching * dA * lambda * mat(0, 0) * aderiv.row(0);
                (*derivative) += kstretching * dA * lambda * mat(0, 1) * aderiv.row(1);
                (*derivative) += kstretching * dA * lambda * mat(1, 0) * aderiv.row(2);
                (*derivative) += kstretching * dA * lambda * mat(1, 1) * aderiv.row(3);
            }

            if (hessian)
            {
                hessian->setZero();

                Eigen::Matrix<double, 1, 9> rankone;
                rankone.setZero();
                rankone += mat(0, 0) * aderiv.row(0);
                rankone += mat(0, 1) * aderiv.row(1);
                rankone += mat(1, 0) * aderiv.row(2);
                rankone += mat(1, 1) * aderiv.row(3);

                (*hessian) += kstretching * dA * rankone.transpose() * rankone;

                (*hessian) += kstretching * dA * lambda * mat(0, 0) * ahess[0];
                (*hessian) += kstretching * dA * lambda * mat(0, 1) * ahess[1];
                (*hessian) += kstretching * dA * lambda * mat(1, 0) * ahess[2];
                (*hessian) += kstretching * dA * lambda * mat(1, 1) * ahess[3];

                //(*hessian) += dA * sign * lambda / denom * (-1.0 / 2.0 / abar.determinant()) * aderiv.row(3).transpose() * aderiv.row(0);
                (*hessian) += kstretching * dA * sign * lambda / denom * (-1.0 / 2.0 * detAbarinv) * aderiv.row(3).transpose() * aderiv.row(0);
                (*hessian) += kstretching * dA * sign * lambda / denom * (-1.0 / 2.0 * detAbarinv) * -1 * aderiv.row(2).transpose() * aderiv.row(1);
                (*hessian) += kstretching * dA * sign * lambda / denom * (-1.0 / 2.0 * detAbarinv) * -1 * aderiv.row(1).transpose() * aderiv.row(2);
                (*hessian) += kstretching * dA * sign * lambda / denom * (-1.0 / 2.0 * detAbarinv) * aderiv.row(0).transpose() * aderiv.row(3);

                Eigen::Matrix<double, 1, 9> abarinvterm;
                abarinvterm.setZero();
                abarinvterm += abarinv(0, 0) * aderiv.row(0);
                abarinvterm += abarinv(0, 1) * aderiv.row(1);
                abarinvterm += abarinv(1, 0) * aderiv.row(2);
                abarinvterm += abarinv(1, 1) * aderiv.row(3);
                (*hessian) += kstretching * dA * sign * lambda / denom / 4.0 * abarinvterm.transpose() * abarinvterm;

                //Eigen::Matrix2d inner = T / 4.0 * abarinv - 1.0 / 2.0 / abar.determinant()  * adjstrain;
                Eigen::Matrix2d inner = T / 4.0 * abarinv - 1.0 / 2.0 * detAbarinv * adjstrain;
                Eigen::Matrix<double, 1, 9> innerVec;
                innerVec.setZero();
                innerVec += inner(0, 0) * aderiv.row(0);
                innerVec += inner(0, 1) * aderiv.row(1);
                innerVec += inner(1, 0) * aderiv.row(2);
                innerVec += inner(1, 1) * aderiv.row(3);
                (*hessian) += kstretching * -dA * sign * lambda / denom / denom / denom * innerVec.transpose() * innerVec;

            }
        }
        return result;
    }
}

template <class SFF>
double TensionFieldStVKMaterial<SFF>::bendingEnergy(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& extraDOFs,
    double thickness,
    const Eigen::Matrix2d& abar, const Eigen::Matrix2d& bbar,
    int face,
    Eigen::Matrix<double, 1, 18 + 3 * SFF::numExtraDOFs>* derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
    Eigen::Matrix<double, 18 + 3 * SFF::numExtraDOFs, 18 + 3 * SFF::numExtraDOFs>* hessian) const
{
    if (derivative)
        derivative->setZero();
    if (hessian)
        hessian->setZero();
    return 0.0;
}

// instantiations
template class TensionFieldStVKMaterial<MidedgeAngleSinFormulation>;
template class TensionFieldStVKMaterial<MidedgeAngleTanFormulation>;
template class TensionFieldStVKMaterial<MidedgeAverageFormulation>;
