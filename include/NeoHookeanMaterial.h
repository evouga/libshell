#ifndef NEOHOOKEANMATERIAL_H
#define NEOHOOKEANMATERIAL_H

#include "MaterialModel.h"

namespace LibShell {

    /*
    * Neo-Hookean nonlinear material model, with energy density
    * W = beta/2.0 (tr[M] - 2 - log det M) + alpha/2.0 (log det M / 2)^2
    * where M = gbar^-1 g, and g and gbar are the current and rest metrics of the
    * shell volume (which vary in the thickness direction as defined by the surface
    * fundamental forms).
    *
    * Takes a MonolayerRestState.
    */

    template <class SFF>
    class NeoHookeanMaterial : public MaterialModel<SFF>
    {
    public:
        NeoHookeanMaterial(double lameAlpha, double lameBeta) : lameAlpha_(lameAlpha), lameBeta_(lameBeta) {}

        /*
        * Lame parameters of the material (as in the energy density written above)
        */
        double lameAlpha_, lameBeta_;

        virtual double stretchingEnergy(
            const MeshConnectivity& mesh,
            const Eigen::MatrixXd& curPos,
            const RestState &restState,
            int face,
            Eigen::Matrix<double, 1, 9>* derivative, // F(face, i)
            Eigen::Matrix<double, 9, 9>* hessian) const;

        virtual double bendingEnergy(
            const MeshConnectivity& mesh,
            const Eigen::MatrixXd& curPos,
            const Eigen::VectorXd& extraDOFs,
            const RestState &restState,
            int face,
            Eigen::Matrix<double, 1, 18 + 3 * SFF::numExtraDOFs>* derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
            Eigen::Matrix<double, 18 + 3 * SFF::numExtraDOFs, 18 + 3 * SFF::numExtraDOFs>* hessian) const;


    };
};

#endif
