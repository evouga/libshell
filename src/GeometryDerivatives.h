#ifndef GEOMETRYDERIVATIVES_H
#define GEOMETRYDERIVATIVES_H

#include <Eigen/Core>
#include <vector>

class MeshConnectivity;

Eigen::Matrix3d crossMatrix(Eigen::Vector3d v);
Eigen::Matrix2d adjugate(Eigen::Matrix2d M);

/*
* Signed angle between two vectors, as measured on the oriented plane with normal axis (which should be perpendicular to both vectors)
* Derivatives are with respect to v then w
*/
double angle(const Eigen::Vector3d &v, const Eigen::Vector3d &w, const Eigen::Vector3d &axis,
    Eigen::Matrix<double, 1, 6> *derivative, // v, w
    Eigen::Matrix<double, 6, 6> *hessian
);

/* 
* Normal vector perpendcular to a face, and with magnitude equal to double the face area.
* Derivatives are with respect to vertices (startidx, startidx+1, startidx+2) of the face (modulo 3)
*/
Eigen::Vector3d faceNormal(const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    int face, int startidx,
    Eigen::Matrix<double, 3, 9> *derivative,
    std::vector<Eigen::Matrix<double, 9, 9> > *hessian);

/*
* Altitude to edge edgeidx.
* Derivatives are with respect to vertices (edgeidx, edgeidx+1, edgeidx+2) of the face (modulo 3)
*/
double triangleAltitude(const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    int face,
    int edgeidx,
    Eigen::Matrix<double, 1, 9> *derivative,
    Eigen::Matrix<double, 9, 9> *hessian);

/* 
 * With respect to the barycentric basis on face.
 * Derivatives are with respect to vertices (0, 1, 2) of the face.
 */
Eigen::Matrix2d firstFundamentalForm(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    int face,
    Eigen::Matrix<double, 4, 9> *derivative, // F(face, i)
    std::vector <Eigen::Matrix<double, 9, 9> > *hessian);

#endif
