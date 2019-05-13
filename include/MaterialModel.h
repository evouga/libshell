#ifndef MATERIALMODEL_H
#define MATERIALMODEL_H

#include <Eigen/Core>

class MeshConnectivity;
class SecondFundamentalFormDiscretization;

class MaterialModel
{
public:
    virtual double stretchingEnergy(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        double thickness,
        const Eigen::Matrix2d &abar,
        int face,
        Eigen::Matrix<double, 1, 9> *derivative, // F(face, i)
        Eigen::Matrix<double, 9, 9> *hessian) const = 0;

    virtual double bendingEnergy(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        const Eigen::VectorXd &extraDOFs,
        double thickness,
        const Eigen::Matrix2d &abar, const Eigen::Matrix2d &bbar,
        int face,
        const SecondFundamentalFormDiscretization &sff,
        Eigen::MatrixXd *derivative, // F(face, i), then the three vertices opposite F(face,i), then the extra DOFs on oppositeEdge(face,i)
        Eigen::MatrixXd *hessian) const = 0;
};

#endif