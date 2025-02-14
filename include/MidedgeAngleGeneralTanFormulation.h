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
 * Second Fundamental Form Based on Midedge Normals
 *
 * The second fundamental form (II) is given as:
 *
 * n = (2 u + 2 v - 1) n0 - (1 - 2 u) n1 - (1 - 2 v) n2,
 * n = m(u, v) d(u, v)
 * m(u, v) = (2 u + 2 v - 1) m0 - (1 - 2 u) m1 - (1 - 2 v) m2,
 * d(u, v) = (2 u + 2 v - 1) d0 - (1 - 2 u) d1 - (1 - 2 v) d2,
 *
 *   II = -2 * [(n1 - n0)^T(x1 - x0), (n1 - n0)^T(x2 - x0);
 *             (n2 - n0)^T(x1 - x0), (n2 - n0)^T(x2 - x0)]
 *
 * where:
 *   - n0, n1, n2: normals of the three edges ei, which are opposite the face vertex i.
 *   - x0, x1, x2: positions of the three vertices of the face.
 *
 * To encode the normal ni, we associate it with:
 *   - A magnitude mi ∈ R+.
 *   - A direction di ∈ S^2.
 *
 * The direction di is represented as a unit vector in the basis {ê, vb, ê x vb}, where:
 *   - vb: the bisector vector of two face normals.
 *   - ê: the normalized edge vector.
 *
 * Thus, di is expressed as:
 *
 *   di = cos(σ) ê + sin(σ) cos(γ) vb + sin(σ) sin(γ) (ê x vb),
 *
 * where:
 *   - σ (sigma): angle between di and ê, in [0, π].
 *   - γ (gamma): rotation angle from vb to di, with ê as the axis, in [0, 2π).
 *
 * We allow differences in the magnitude (mi) but not the direction (di). That is:
 *   - We use half-edge CR for the magnitude.
 *   - CR is used for the direction.
 *
 * As a result, each edge has four degrees of freedom (DOFs):
 *   - Two angles: γ (gamma) and σ (sigma).
 *   - Two magnitudes: m0 and m1 (for each half-edge).
 *
 * **Rewriting the Formula:**
 * For each face, we can rewrite the equation under the face normal basis nf as:
 *
 *   di = cos(σ) ê + sin(σ) cos(ζ) nf + sin(σ) sin(ζ) (ê x nf),
 *
 * where:
 *   - ζ (zeta) = γ + s * θ / 2.
 *   - s ∈ {1, -1} represents the edge orientation.
 *   - θ (theta) is the dihedral angle.
 *
 * In other words, ζ is the rotation angle from di to nf with ê as the rotation axis.
 *
 * Denote:
 *   - b0 = x1 - x0 (parallel to e2).
 *   - b1 = x2 - x0 (parallel to e1).
 *
 * Then:
 *   ni^T bj = mi cos(σ) (bj^T ei) / |ei| + mi sin(σ) sin(ζ) s_ij * |bj x ei| / |ei|,
 *           = mi cos(σ) (bj^T ei) / |ei| + mi sin(σ) sin(ζ) s_ij * h_i (if bj is not parallel to ei),
 *           = mi cos(σ) |ei| * sign(bj^T ei) (if bj is parallel to ei),
 *
 *
 * where:
 *   - s_ij = sign(nf ⋅ (bj x ei)) (assumes no triangle flip).
 *   - h_i: altitude of the triangle on edge ei.
 *
 * Precomputations for s_ij and sign(bj^T ei) can be done to simplify calculations.
 *
 * **Symmetry in II:**
 * For general cases, the following symmetry condition does not hold:
 *
 *   (n1 - n0)^T(x2 - x0) = (n2 - n0)^T(x1 - x0).
 *
 * To ensure symmetry in II, we redefine it as:
 *
 *   II = -2 * [(n1 - n0)^T(x1 - x0), 1/2 ((n1 - n0)^T(x2 - x0) + (n2 - n0)^T(x1 - x0));
 *             1/2 ((n1 - n0)^T(x2 - x0) + (n2 - n0)^T(x1 - x0)), (n2 - n0)^T(x2 - x0)].
 *
 * This corresponds to defining II as:
 *
 *   II = 1/2 (dn^T dr + dr^T dn),
 *
 * in the smooth setting.
 *
 * ** Tan-Formulation: **
 * In the tan-formulation, we treat the edge normals are always unit along the face normal direction, in other words the
 * extruded prism has the constant thickness, that is mi di^nf = 1, leading to
 *  - mi = 1 / (sin(σ) cos(ζ))
 *  - two degrees of freedom for each edge: Two angles: γ (gamma) and σ (sigma).
 *
 * and corresponding ni^^ bj is given as:
 *   ni^T bj = cot(σ) / cos(ζ) (bj^T ei) / |ei| + tan(ζ) s_ij * |bj x ei| / |ei|,
 *           = cot(σ) / cos(ζ) (bj^T ei) / |ei| + tan(ζ) s_ij * h_i (if bj is not parallel to ei),
 *           = cot(σ) / cos(ζ) |ei| * sign(bj^T ei) (if bj is parallel to ei),
 */

class MidedgeAngleGeneralTanFormulation {
public:
    constexpr static int numExtraDOFs = 2;

    /*
     * Initialize the extra dofs, as well as the sign sij,
     *
     * @param[in] extraDOFs:        the extra degrees of freedom of on the edges
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex positions
     */
    static void initializeExtraDOFs(Eigen::VectorXd& extraDOFs,
                                    const MeshConnectivity& mesh,
                                    const Eigen::MatrixXd& curPos);

    /*
     * Compute the second fundamental form for the specific face
     *
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex position
     * @param[in] extraDOFs:        the current edge dofs
     * @param[in] face:             the face id, where we want to compute the second fundamental form
     *
     * @param[out] derivative:      the derivative of each entry of the second fundamental form
     * @param[out] hessian:         the hessian of each entry of the second fundamental form
     *
     * @return:                     the second fundamental form
     *
     */
    static Eigen::Matrix2d secondFundamentalForm(
        const MeshConnectivity& mesh,
        const Eigen::MatrixXd& curPos,
        const Eigen::VectorXd& extraDOFs,
        int face,
        Eigen::Matrix<double, 4, 18 + 3 * numExtraDOFs>*
            derivative,  // F(face, i), then the three vertices opposite F(face,i), then the thetas on
                         // oppositeEdge(face,i)
        std::vector<Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>>* hessian);

    /*
     * Test function of ni^Tbj computation
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex position
     * @param[in] extraDOFs:        the current edge dofs
     * @param[in] face:             the face id, where we want to compute the second fundamental form
     * @param[in] i:                the index for edge normal, in {0, 1, 2}
     * @param[in] j:                the index for basis, in {0, 1}
     */
    static void test_compute_nibj(const MeshConnectivity& mesh,
                                  const Eigen::MatrixXd& curPos,
                                  const Eigen::VectorXd& edgeDOFs,
                                  int face,
                                  int i,
                                  int j);

    /*
     * Test function of second fundamental form computation
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex position
     * @param[in] extraDOFs:        the current edge dofs
     * @param[in] face:             the face id, where we want to compute the second fundamental form
     */
    static void test_second_fund_form(const MeshConnectivity& mesh,
                                      const Eigen::MatrixXd& curPos,
                                      const Eigen::VectorXd& edgeDOFs,
                                      int face);

    /*
     * Get per edge face sigma and zeta, for the general tan formulation, the magnitude mi = 1 / (sin(σ) cos(ζ))
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex position
     * @param[in] extraDOFs:        the current edge dofs
     * @param[in] edge:             the edge id
     * @param[in] face:             the face id in {0, 1}
     *
     * @param[out] sigma:           the angle between the edge direction d and the edge e
     * @param[out] zeta:            the rotation angle from face normal nf to edge direction d with e as the rotation
     * axis
     */
    static void get_per_edge_face_sigma_zeta(const MeshConnectivity& mesh,
                                             const Eigen::MatrixXd& curPos,
                                             const Eigen::VectorXd& edgeDOFs,
                                             int edge,
                                             int face,
                                             double& sigma,
                                             double& zeta);

    static std::vector<Eigen::Vector3d> get_face_edge_normals(const MeshConnectivity& mesh,
                                                              const Eigen::MatrixXd& curPos,
                                                              const Eigen::VectorXd& edgeDOFs,
                                                              int face);

public:
    /*
     * The relationships between two vectors with the axis face normal
     */
    enum class VectorRelationship {
        kSameDirection = 0,
        kOppositeDirection = 1,
        kPositiveOrientation = 2,
        kNegativeOrientation = 3,
        kUndefined = 4
    };

    /*
     * Get relationship between the edge and face basis (two adjacent faces)
     *
     * @param[in] mesh:             the mesh connectivity
     * @param[in] eid:              the edge id
     */
    static std::vector<VectorRelationship> get_edge_face_basis_relationship(const MeshConnectivity& mesh, int eid);

    /*
     * Compute ni^T bj
     *
     *
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex position
     * @param[in] extraDOFs:        the current edge dofs
     * @param[in] face:             the face id, where we want to compute the second fundamental form
     * @param[in] i:                the index for edge normal, in {0, 1, 2}
     * @param[in] j:                the index for basis, in {0, 1}
     *
     * @param[out] derivative:      the derivative of ni^T bj
     * @param[out] hessian:         the hessian of ni^T bj
     *
     * @return:                     ni^T bj
     *
     * @note:
     *   ni^T bj = cot(σ) / cos(ζ) (bj^T ei) / |ei| - tan(ζ) s_ij * |bj x ei| / |ei|,
     *           = cot(σ) / cos(ζ) (bj^T ei) / |ei| - tan(ζ) s_ij * h_i (if bj is not parallel to ei),
     *           = cot(σ) / cos(ζ) |ei| * sign(bj^T ei) (if bj is parallel to ei),
     */
    static double compute_nibj(const MeshConnectivity& mesh,
                               const Eigen::MatrixXd& curPos,
                               const Eigen::VectorXd& edgeDOFs,
                               int face,
                               int i,
                               int j,
                               Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>* derivative,
                               Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>* hessian);

    /*
     * Compute ni^T ei
     *
     *
     * @param[in] mesh:             the mesh connectivity
     * @param[in] curPos:           the current vertex position
     * @param[in] extraDOFs:        the current edge dofs
     * @param[in] face:             the face id, where we want to compute the second fundamental form
     * @param[in] i:                the index for edge normal, in {0, 1, 2}
     *
     * @param[out] derivative:      the derivative of ni^T ei
     * @param[out] hessian:         the hessian of ni^T ei
     *
     * @return:                     ni^T ei
     *
     * @note:
     * ni^T ei = mi cos(sigma_i) |ei|
     */
    static double compute_niei(const MeshConnectivity& mesh,
                               const Eigen::MatrixXd& curPos,
                               const Eigen::VectorXd& edgeDOFs,
                               int face,
                               int i,
                               Eigen::Matrix<double, 1, 18 + 3 * numExtraDOFs>* derivative,
                               Eigen::Matrix<double, 18 + 3 * numExtraDOFs, 18 + 3 * numExtraDOFs>* hessian);

};
};  // namespace LibShell
