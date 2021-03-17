#ifndef BILAYERSTVKMATERIAL_H
#define BILAYERSTVKMATERIAL_H

#include "MaterialModel.h"

namespace LibShell {

    /*
     * St. Venant-Kirchhoff linear material model, with energy density
     * W = alpha/2.0 tr(S)^2 + beta tr(S^2),
     * for a bilayer where each layer has a different thickness, rest metric, and
     * Lame parameters. As in the monolayer StVK formulation, the strain tensor
     * for each layer is S_i = gbar_i^{-1}(g_i-gbar_i), where g_i and gbar_i are
     * the current and rest metrics of the shell volume within layer i (which vary
     * in the thickness direction as defined by the surface fundamental forms).
     *
     * Takes a BilayerRestState.
     */

    template <class SFF>
    class BilayerStVKMaterial : public MaterialModel<SFF>
    {
    public:
        BilayerStVKMaterial(double lameAlpha1, double lameBeta1,
            double lameAlpha2, double lameBeta2
        )
            : lameAlpha1_(lameAlpha1), lameBeta1_(lameBeta1),
            lameAlpha2_(lameAlpha2), lameBeta2_(lameBeta2)
        {}

        /*
         * Lame parameters of the material (as in the energy density written above)
         */
        double lameAlpha1_, lameBeta1_;
        double lameAlpha2_, lameBeta2_;

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