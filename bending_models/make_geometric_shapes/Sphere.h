#ifndef SPHERE_H
#define SPHERE_H

#include <Eigen/Core>

void makeSphere(double radius, double triangleArea,Eigen::MatrixXd &V, Eigen::MatrixXi &F); 
void sphereFromSamples(int samples, Eigen::MatrixXd &V, Eigen::MatrixXi &F);

#endif
