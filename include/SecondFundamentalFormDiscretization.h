#ifndef SECONDFUNDAMENTALFORMDISCRETIZATION_H
#define SECONDFUNDAMENTALFORMDISCRETIZATION_H

#include <Eigen/Core>
#include <vector>

class MeshConnectivity;

class SecondFundamentalFormDiscretization
{
public:
    // Number of extra DOFs that the simulation needs to track per edge (angle, directors, etc)
    virtual int numExtraDOFs() const = 0;

    virtual void initializeExtraDOFs(Eigen::VectorXd &extraDOFs, const MeshConnectivity &mesh, const Eigen::MatrixXd &curPos) const = 0;

    /*
    * Second fundamental form in the barycentric basis of face.
    * Derivatives are with respect to vertices (0, 1, 2) of face, then vertices opposite (0, 1, 2), then the extra DOFs
    */
    virtual Eigen::Matrix2d secondFundamentalForm(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        const Eigen::VectorXd &extraDOFs,
        int face,
        Eigen::MatrixXd *derivative, 
        std::vector<Eigen::MatrixXd> *hessian) const = 0;
};

#endif