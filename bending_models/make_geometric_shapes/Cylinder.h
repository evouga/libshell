#ifndef CYLINDER_H
#define CYLINDER_H

#include <Eigen/Core>
#include <vector>


void makeCylinder(bool regular, double radius, double height, double triangleArea,
    Eigen::MatrixXd& flatV,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    double angle = 2 * M_PI);

void makeTwistedCylinderWithoutSeam(bool regular, double radius, double height, double triangleArea,
    Eigen::MatrixXd& flatV,
    Eigen::MatrixXi& flatF,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F, 
    double twist_angle = 0,
    std::vector<int>* flatV_to_V_map = nullptr);


void getBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<int>& topVertices, std::vector<int>& bottomVertices);

#endif