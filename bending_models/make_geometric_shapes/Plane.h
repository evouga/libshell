#pragma once

#include <Eigen/Core>
#include <vector>


void makePlane(bool regular, double W, double H, double triangleArea,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);
