#ifndef CYLINDER_H
#define CYLINDER_H

#include <Eigen/Core>
#include <vector>

void makeCylinder(double radius, double height, double triangleArea,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F);


void getBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<int>& topVertices, std::vector<int>& bottomVertices);

#endif