#ifndef MIDEDGEAVERAGEFORMULATION_H
#define MIDEDGEAVERAGEFORMULATION_H

#include "SecondFundamentalFormDiscretization.h"

class MidedgeAverageFormulation : public SecondFundamentalFormDiscretization
{
public:
    virtual int numExtraDOFs() const;

    virtual void initializeExtraDOFs(Eigen::VectorXd &extraDOFs, const MeshConnectivity &mesh, const Eigen::MatrixXd &curPos) const;

    virtual Eigen::Matrix2d secondFundamentalForm(
        const MeshConnectivity &mesh,
        const Eigen::MatrixXd &curPos,
        const Eigen::VectorXd &extraDOFs,
        int face,
        Eigen::MatrixXd *derivative, // F(face, i), then the three vertices opposite F(face,i), then the thetas on oppositeEdge(face,i)
        std::vector<Eigen::MatrixXd > *hessian) const;
};

#endif