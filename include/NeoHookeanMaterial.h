#ifndef NEOHOOKEANMATERIAL_H
#define NEOHOOKEANMATERIAL_H

#include "MaterialModel.h"

/*
* Neo-Hookean nonlinear material model, with energy density
* W = alpha/2.0 (tr[gbar^{-1} g]/sqrt(det[gbar^{-1}g]) - 2) + beta/2.0 (sqrt(det[gbar^{-1}g]) - 1)^2
* where g and gbar are the current and rest metrics of the shell volume (which 
* vary in the thickness direction as defined by the surface fundamental forms).
*/

class NeoHookeanMaterial : public MaterialModel
{
public:
    NeoHookeanMaterial(double lameAlpha, double lameBeta) : lameAlpha_(lameAlpha), lameBeta_(lameBeta) {}

    /*
    * Lame parameters of the material (as in the energy density written above)
    */
    double lameAlpha_, lameBeta_;

    virtual double stretchingEnergy(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        double thickness,
        const Eigen::Matrix2d &abar,
        int face,
        Eigen::Matrix<double, 1, 9> *derivative, // F(face, i)
        Eigen::Matrix<double, 9, 9> *hessian) const;

    virtual double bendingEnergy(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        const Eigen::VectorXd &extraDOFs,
        double thickness,
        const Eigen::Matrix2d &abar, const Eigen::Matrix2d &bbar,
        int face,
        const SecondFundamentalFormDiscretization &sff,
        Eigen::MatrixXd *derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
        Eigen::MatrixXd *hessian) const;


};


#endif