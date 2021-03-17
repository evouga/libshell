#ifndef STVKMATERIAL_H
#define STVKMATERIAL_H

#include "MaterialModel.h"

namespace LibShell {

    /*
     * St. Venant-Kirchhoff linear material model, with energy density
     * W = alpha/2.0 tr(S)^2 + beta tr(S^2),
     * for strain tensor S = gbar^{-1}(g-gbar), where g and gbar are the current
     * and rest metrics of the shell volume (which vary in the thickness direction
     * as defined by the surface fundamental forms).
     *
     * Needs a MonolayerRestState.
     */

    template <class SFF>
    class StVKMaterial : public MaterialModel<SFF>
    {
    public:
        StVKMaterial(double lameAlpha, double lameBeta) : lameAlpha_(lameAlpha), lameBeta_(lameBeta) {}

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