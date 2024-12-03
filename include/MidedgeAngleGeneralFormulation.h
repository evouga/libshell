//
// Created by Zhen Chen on 11/25/24.
//
#pragma once

#pragma once

#include "MeshConnectivity.h"

#include <Eigen/Core>
#include <vector>

namespace LibShell {

class MeshConnectivity;

/*
 * Second fundamental form based on midedge normals.
 *
 * The actually is given as:
 * II = 2 [(n1 - n0)^T(x1 - x0), (n1 - n0)^T(x2 - x0); (n2 - n0)^T(x1 - x0), (n2 - n0)^T(x2 - x0)]
 * where n0, n1, n2 are the normals of the three edges ei, which is the oppiste face of vertex i;
 * and x0, x1, x2 are the positions of the three vertices of the face.
 *
 * To encode the normal ni, we associate it with a magnitude mi \in R+, and a direction di \in S^2
 * di is represented as a unit vector in {vb, ehat, vb x ehat}, where vb is the bisector vector of two face normals,
 * and ehat is normalized the edge vector.
 * that is,
 *
 * (1) di = cos(sigma) ehat + sin(sigma) cos(gamma) vb + sin(sigma) sin(gamma) (vb x ehat)
 *
 * We only allow the disagreement in the magnitude, but not the direction. (that is we use half-edge CR for magnitude,
 * and CR for direction) Therefore, we have 4 DOFs for each edge, with first two are two angles, gamma and sigma, and
 * the last two are the magnitudes (for each half edge).
 *
 * Formula rewritten:
 * For each face, we can rewrite the equation as under the face normal nf basis as:
 *
 * (2) di = cos(sigma) ehat + sin(sigma) cos(zeta) nf + sin(sigma) sin(zeta) (nf x ehat)
 *
 * where zeta = s * gamma + theta/2, with s in {1, -1} indicating the orientation of the edge, and theta is dihedral
 * angle. Denote b0 = x1 - x0 (parallel to e2), and b1 = x2 - x0 (parallel to e1), we have
 *
 * ni^T bj = mi |bj| cos(sigma) cos(theta_ji) - mi |bj| sin(sigma) sin(zeta) sin(theta_ji)
 * where theta_ji is the rotation angle from bj to ei, with axis nf,
 *
 * This can be further rewritten as
 * ni^T bj = mi cos(sigma) bj^T ei / |ei| - mi sin(sigma) sin(zeta) sij * |bj x ei| / |ei|
 *         = mi cos(sigma) bj^T ei / |ei| - mi sin(sigma) sin(zeta) sij * hi, if bj is not parallel to ei
 *         = mi cos(sigma) |ei| * sign(bj^T ei), if bj is parallel to ei
 *
 * where sij = sign(nf.dot(bj x ei)). Assume there is not triangle flip, sij can be precomputed, same as sign(bj^T ei)
 * and hi is the altitude of the triangle on the edge ei.
 */

class MidedgeAngleGeneralFormulation {
public:
    constexpr static int numExtraDOFs = 4;

    static void initializeExtraDOFs(Eigen::VectorXd& extraDOFs,
                                    const MeshConnectivity& mesh,
                                    const Eigen::MatrixXd& curPos);

    static Eigen::Matrix2d secondFundamentalForm(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        int face,
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs>*
            derivative,  // F(face, i), then the three vertices opposite F(face,i), then the thetas on
                         // oppositeEdge(face,i)
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>>* hessian);

    enum class VectorRelationship{
        kSameDirection = 0,
        kOppositeDirection = 1,
        kPositiveOrientation = 2,
        kNegativeOrientation = 3,
        kUndefined = 4
    };

    // ni^T bj = mi cos(sigma_i) bj^T ei / |ei| - mi sin(sigma_i) sin(zeta_i) sij * |bj x ei| / |ei|
    //         = mi cos(sigma_i) bj^T ei / |ei| - mi sin(sigma_i) sin(zeta_i) sij * hi, if bj is not parallel to ei
    //         = mi cos(sigma_i) |ei| * sign(bj^T ei), if bj is parallel to ei
    static double compute_nibj(const MeshConnectivity& mesh,
                               const Eigen::MatrixXd& curPos,
                               const Eigen::VectorXd& edgeDOFs,
                               int face,
                               int i,
                               int j,
                               Eigen::Matrix<double, 1, 30>* derivative,
                               Eigen::Matrix<double, 30, 30>* hessian);
public:
    static void test_compute_nibj(const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos, const Eigen::VectorXd& edgeDOFs, int face, int i, int j);
    static void test_second_fund_form(const MeshConnectivity& mesh, const Eigen::MatrixXd& curPos, const Eigen::VectorXd& edgeDOFs, int face);

public:
    static std::vector<std::array<VectorRelationship, 2>> m_edge_face_basis_sign;
};
};  // namespace LibShell
