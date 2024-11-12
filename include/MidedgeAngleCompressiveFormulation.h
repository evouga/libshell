#pragma once

#include <Eigen/Core>
#include <vector>

namespace LibShell {

class MeshConnectivity;

/*
 * Second fundamental form based on rotating the averaged face normals across edges by an addition per-edge angle.
 * For each edge, we assign an extra degree of freedom to represent its norm on the face, which can vary across the faces
 */

class MidedgeAngleCompressiveFormulation {
public:
    constexpr static int numExtraDOFs = 3;

    static void initializeExtraDOFs(Eigen::VectorXd& extraDOFs,
                                    const MeshConnectivity& mesh,
                                    const Eigen::MatrixXd& curPos);

    static Eigen::Matrix2d secondFundamentalForm(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        int face,
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs>*
            derivative,  // F(face, i), then the three vertices opposite F(face,i), then the thetas on
                         // oppositeEdge(face,i)
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>>* hessian);
};
};  // namespace LibShell
