#ifndef HALFCYLINDER_H
#define HALFCYLINDER_H

#include <Eigen/Core>
#include <vector>

void makeHalfCylinder(bool regular, double radius, double height, double triangleArea,
    Eigen::MatrixXd& flatV,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);


void getBoundaries(const Eigen::MatrixXi& F, std::vector<int>& bdryVertices);

#endif