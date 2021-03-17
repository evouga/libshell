#ifndef MIDEDGEAVERAGEFORMULATION_H
#define MIDEDGEAVERAGEFORMULATION_H

#include <Eigen/Core>
#include <vector>

namespace LibShell {

    class MeshConnectivity;

    class MidedgeAverageFormulation
    {
    public:
        constexpr static int numExtraDOFs = 0;

        static void initializeExtraDOFs(Eigen::VectorXd& extraDOFs, const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos);

        static Eigen::Matrix2d secondFundamentalForm(
            const MeshConnectivity& mesh,
            const Eigen::MatrixXd& curPos,
            const Eigen::VectorXd& extraDOFs,
            int face,
            Eigen::Matrix<double, 4, 18>* derivative, // F(face, i), then the three vertices opposite F(face,i), then the thetas on oppositeEdge(face,i)
            std::vector<Eigen::Matrix<double, 18, 18> >* hessian);
    };
};

#endif