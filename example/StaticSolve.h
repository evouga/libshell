#ifndef STATICSOLVE_H
#define STATICSOLVE_H

#include <Eigen/Core>
#include <vector>

class MeshConnectivity;
class MaterialModel;
class SecondFundamentalFormDiscretization;

void takeOneStep(const MeshConnectivity &mesh,
    Eigen::MatrixXd &curPos,
    Eigen::VectorXd &curEdgeDOFs,
    const MaterialModel &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    const std::vector<Eigen::Matrix2d> &bbars,
    const SecondFundamentalFormDiscretization &sff,
    double &reg);

#endif