#ifndef ELASTICSHELL_H
#define ELASTICSHELL_H

#include <Eigen/Core>
#include <vector>
#include <Eigen/Sparse>
#include "MaterialModel.h"

namespace LibShell {

    class MeshConnectivity;
    struct RestState;

    template <class SFF>
    class ElasticShell
    {
    public:
        /*
         * Computes the elastic energy of a shell and, optionally, the derivative and Hessian of the shell elastic energy.
         *
         * Inputs:
         * - mesh:          data structure encoding connectivity information about the mesh. This data structure is used instead of a raw list
                            of faces so that (expensive) computation of connectivity data structures does not need to be repeated with every call.
         * - curPos:        |V| x 3 matrix of the current positions of the mesh vertices.
         * - edgeDOFs:      extra degrees of freedom needed depending on the choice of second fundamental form discretization (director angles, e.g.).
         * - mat:           Material model (e.g. StVKMaterial for St. Venant-Kirchhoff) defining the elastic energy density of the shell.
         * - restState:     A RestState container encoding the geometry (thicknesses, rest curvature, etc.) of the shell's strain-free state. Different
         *                  material models will require different rest state data.
         * - SFF:           the choice of second fundamental form discretization.
         * - whichTerms     optional flags offering finer-grained control over which terms to include. ET_STRETCHING includes the bending energy, and
                            ET_BENDING the bending energy. Default is both (ET_STRETCHING | ET_BENDING).
         *
         * Outputs:
         * - returns the total elastic energy of the shell.
         * - derivative:    if not null, will be set to the derivative (*negative* of the force) of the elastic energy. The derivative is
         *                  flattened: entry 3*i+j corresponds to the derivative of energy with respect to coordinate j of the ith vertex. Following the
         *                  vertex derivatives are the derivative with respect to the extra degrees of freedom edgeDOFs (if any).
         * - hessian:       if not null, will be set to the (sparse) Hessian of the elastic energy. The indexing scheme for the Hessian is the same as for
         *                  the derivative.
         */
        static double elasticEnergy(
            const MeshConnectivity& mesh,
            const Eigen::MatrixXd& curPos,
            const Eigen::VectorXd& edgeDOFs,
            const MaterialModel<SFF>& mat,
            const RestState &restState,
            Eigen::VectorXd* derivative, // positions, then thetas
            std::vector<Eigen::Triplet<double> >* hessian);

        static double elasticEnergy(
            const MeshConnectivity& mesh,
            const Eigen::MatrixXd& curPos,
            const Eigen::VectorXd& edgeDOFs,
            const MaterialModel<SFF>& mat,
            const RestState &restState,
            int whichTerms,
            Eigen::VectorXd* derivative, // positions, then thetas
            std::vector<Eigen::Triplet<double> >* hessian);

        /*
         * Computes current fundamental forms for a given mesh. Can be used to initialize these forms from a given mesh rest state.
         */
        static void firstFundamentalForms(const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos, std::vector<Eigen::Matrix2d>& abars);

        static void secondFundamentalForms(const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos, const Eigen::VectorXd& edgeDOFs, std::vector<Eigen::Matrix2d>& bbars);

        enum EnergyTerm
        {
            ET_STRETCHING = 1,
            ET_BENDING = 2
        };
    };
};
#endif
