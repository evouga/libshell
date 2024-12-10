#include "ShellEnergy.h"

#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "../include/StVKMaterial.h"
#include "../include/ExtraEnergyTerms.h"

#include "../Optimization/include/NewtonDescent.h"
#include "igl/null.h"

#include "make_geometric_shapes/HalfCylinder.h"
#include "make_geometric_shapes/Cylinder.h"
#include "make_geometric_shapes/Sphere.h"

#include <polyscope/surface_vector_quantity.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <igl/principal_curvature.h>
#include <set>
#include <vector>

double cokeRadius;
double cokeHeight;
double sphereRadius;

double thickness;
double poisson;
double youngs;
double shear;

int quad_order;

double triangleArea;

double QBEnergy;

enum class MeshType {
    MT_CYLINDER_IRREGULAR,
    MT_CYLINDER_REGULAR,
    MT_SPHERE,
    MT_FOLD_MESH,
};

MeshType curMeshType;

polyscope::SurfaceMesh* surface_mesh;
polyscope::PointCloud* pt_mesh;

void lameParameters(double& alpha, double& beta) {
    alpha = youngs * poisson / (1.0 - poisson * poisson);
    beta = youngs / 2.0 / (1.0 + poisson);
}

struct Energies {
    double exact;              // the theoretic II
    double quadratic_bending;  // quadratic bending
    double stvk;               // stvk bending with fixed edge normal (average adjacent face normals)
    double stvk_s1_dir_sin;    // stvk bending with s1 direction, and sin formulation
    double stvk_s1_dir_tan;    // stvk bending with s1 direction, and tan formulation
    double stvk_s2_dir_sin;    // stvk bending with s2 direction, and sin formulation
    double stvk_s2_dir_tan;    // stvk bending with s2 direction, and tan formulation
    double stvk_general_dir;   // stvk bending with general direction
};

enum class SFFGeneralType {
    kGeneralFormulation = 0,
    kS2SinFormulation = 1,
    kS2TanFormulation = 2,
};

std::vector<Eigen::Vector3d> get_face_edge_normal_vectors(const Eigen::MatrixXd& cur_pos, const LibShell::MeshConnectivity& mesh, const Eigen::VectorXd& edge_dofs, SFFGeneralType& sff_general_type) {
    std::vector<Eigen::Vector3d> face_edge_normals = {};
    int nfaces = mesh.nFaces();

    for(int i = 0; i < nfaces; i++) {
        switch (sff_general_type) {
            case SFFGeneralType::kGeneralFormulation: {
                std::vector<Eigen::Vector3d> general_edge_normals = LibShell::MidedgeAngleGeneralFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
                for(int j = 0; j < 3; j++) {
                    face_edge_normals.push_back(general_edge_normals[j]);
                }
                break;
            }
            case SFFGeneralType::kS2SinFormulation: {
                std::vector<Eigen::Vector3d> general_edge_normals = LibShell::MidedgeAngleGeneralSinFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
                for(int j = 0; j < 3; j++) {
                    face_edge_normals.push_back(general_edge_normals[j]);
                }
                break;
            }
            case SFFGeneralType::kS2TanFormulation: {
                std::vector<Eigen::Vector3d> general_edge_normals = LibShell::MidedgeAngleGeneralTanFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
                for(int j = 0; j < 3; j++) {
                    face_edge_normals.push_back(general_edge_normals[j]);
                }
                break;
            }
            default:
                break;
        }
    }
    return face_edge_normals;
}

std::vector<Eigen::Vector3d> get_midsurf_face_edge_normal_vectors(const Eigen::MatrixXd& cur_pos, const LibShell::MeshConnectivity& mesh) {
    std::vector<Eigen::Vector3d> face_edge_normals = {};
    int nfaces = mesh.nFaces();

    for(int i = 0; i < nfaces; i++) {
        Eigen::Vector3d b0 = cur_pos.row(mesh.faceVertex(i, 1)) - cur_pos.row(mesh.faceVertex(i, 0));
        Eigen::Vector3d b1 = cur_pos.row(mesh.faceVertex(i, 2)) - cur_pos.row(mesh.faceVertex(i, 0));
        Eigen::Vector3d nf = b0.cross(b1);
        nf.normalize();
        for(int j = 0; j < 3; j++) {
            face_edge_normals.push_back(nf);
        }
    }
    return face_edge_normals;
}

double edgeMagnitudeCompression(const Eigen::VectorXd& edge_dofs,
                                const Eigen::VectorXd& edge_area,
                                double youngs_modulus,
                                double thickness,
                                Eigen::VectorXd* deriv,
                                std::vector<Eigen::Triplet<double>>* hessian) {
    int nedges = edge_area.size();
    if (deriv) {
        deriv->resize(edge_dofs.size());
        deriv->setZero();
    }

    if (hessian) {
        hessian->clear();
    }

    if (edge_dofs.size() == nedges || edge_dofs.size() % nedges != 0) return 0.0;
    int ndof_per_edge = edge_dofs.size() / nedges;

    if (ndof_per_edge != 4) {
        return 0;
    }

    double compression = 0.0;
    for (int i = 0; i < nedges; i++) {
        double mi0 = edge_dofs(i * ndof_per_edge + ndof_per_edge - 1);
        double mi1 = edge_dofs(i * ndof_per_edge + ndof_per_edge - 2);

        compression += (mi0 - 1) * (mi0 - 1) * youngs_modulus * thickness / 4 * edge_area[i];
        compression += (mi1 - 1) * (mi1 - 1) * youngs_modulus * thickness / 4 * edge_area[i];

        if (deriv) {
            (*deriv)(i* ndof_per_edge + ndof_per_edge - 1) +=
                2 * (mi0 - 1) * youngs_modulus * thickness / 4 * edge_area[i];
            (*deriv)(i* ndof_per_edge + ndof_per_edge - 2) +=
                2 * (mi1 - 1) * youngs_modulus * thickness / 4 * edge_area[i];
        }

        if (hessian) {
            hessian->push_back({i * ndof_per_edge + ndof_per_edge - 1, i * ndof_per_edge + ndof_per_edge - 1,
                                2 * youngs_modulus * thickness / 4 * edge_area[i]});
            hessian->push_back({i * ndof_per_edge + ndof_per_edge - 2, i * ndof_per_edge + ndof_per_edge - 2,
                                2 * youngs_modulus * thickness / 4 * edge_area[i]});
        }
    }
    return compression;
}

double edgeDOFPenaltyEnergy(const Eigen::VectorXd& edgeDOFs,
                            const Eigen::VectorXd& edge_area,
                            double penaltyScale,
                            Eigen::VectorXd* deriv,
                            std::vector<Eigen::Triplet<double>>* hessian) {
    int nedges = edge_area.size();
    if (deriv) {
        deriv->resize(edgeDOFs.size());
        deriv->setZero();
    }

    if (hessian) {
        hessian->clear();
    }

    if (edgeDOFs.size() == nedges || edgeDOFs.size() % nedges != 0) return 0.0;
    int ndof_per_edge = edgeDOFs.size() / nedges;

    double penalty = 0.0;
    for (int i = 0; i < nedges; i++) {
        penalty +=
            (edgeDOFs(i * ndof_per_edge + 1) - 1) * (edgeDOFs(i * ndof_per_edge + 1) - 1) * penaltyScale * edge_area[i];
        penalty +=
            (edgeDOFs(i * ndof_per_edge + 2) - 1) * (edgeDOFs(i * ndof_per_edge + 2) - 1) * penaltyScale * edge_area[i];

        if (deriv) {
            (*deriv)(i* ndof_per_edge + 1) += 2 * (edgeDOFs(i * ndof_per_edge + 1) - 1) * penaltyScale * edge_area[i];
            (*deriv)(i* ndof_per_edge + 2) += 2 * (edgeDOFs(i * ndof_per_edge + 2) - 1) * penaltyScale * edge_area[i];
        }

        if (hessian) {
            hessian->push_back({i * ndof_per_edge + 1, i * ndof_per_edge + 1, 2 * penaltyScale * edge_area[i]});
            hessian->push_back({i * ndof_per_edge + 2, i * ndof_per_edge + 2, 2 * penaltyScale * edge_area[i]});
        }
    }
    return penalty;
}

// shear * h / 2 * ||n^T e||^2
double edgeVectorPerpTerm(const Eigen::MatrixXd& cur_pos,
                          const Eigen::VectorXd& edge_dofs,
                          const LibShell::MeshConnectivity& mesh,
                          const Eigen::VectorXd& edge_area,
                          SFFGeneralType general_type,
                          Eigen::VectorXd* derivative,
                          std::vector<Eigen::Triplet<double>>* hessian,
                          bool is_proj) {
    assert(mesh.nEdges() == edge_area.size());
    int ndofs_per_edge = edge_dofs.size() / mesh.nEdges();
    int nverts = cur_pos.rows();

    double energy = 0;

    if (derivative) {
        derivative->setZero(3 * nverts + ndofs_per_edge * mesh.nEdges());
    }

    if (hessian) {
        hessian->clear();
    }

    for (int i = 0; i < mesh.nFaces(); i++) {
        for (int j = 0; j < 3; j++) {
            Eigen::VectorXd local_deriv;
            Eigen::MatrixXd local_hessian;
            double w = shear * thickness * 2;

            switch (general_type) {
                case SFFGeneralType::kGeneralFormulation: {
                    assert(ndofs_per_edge == LibShell::MidedgeAngleGeneralFormulation::numExtraDOFs);
                    Eigen::Matrix<double, 1, 18 + 3 * LibShell::MidedgeAngleGeneralFormulation::numExtraDOFs> grad;
                    Eigen::Matrix<double, 18 + 3 * LibShell::MidedgeAngleGeneralFormulation::numExtraDOFs,
                                  18 + 3 * LibShell::MidedgeAngleGeneralFormulation::numExtraDOFs>
                        hess;
                    double niei = LibShell::MidedgeAngleGeneralFormulation::compute_niei(
                        mesh, cur_pos, edge_dofs, i, j, derivative ? &grad : nullptr, hessian ? &hess : nullptr);

                    energy += w * niei * niei;

                    local_deriv = 2 * w * niei * grad.transpose();
                    local_hessian = 2 * (w * grad.transpose() * grad + w * niei * hess);
                    break;
                }

                case SFFGeneralType::kS2SinFormulation: {
                    assert(ndofs_per_edge == LibShell::MidedgeAngleGeneralSinFormulation::numExtraDOFs);
                    Eigen::Matrix<double, 1, 18 + 3 * LibShell::MidedgeAngleGeneralSinFormulation::numExtraDOFs> grad;
                    Eigen::Matrix<double, 18 + 3 * LibShell::MidedgeAngleGeneralSinFormulation::numExtraDOFs,
                                  18 + 3 * LibShell::MidedgeAngleGeneralSinFormulation::numExtraDOFs>
                        hess;
                    double niei = LibShell::MidedgeAngleGeneralSinFormulation::compute_niei(
                        mesh, cur_pos, edge_dofs, i, j, derivative ? &grad : nullptr, hessian ? &hess : nullptr);

                    energy += w * niei * niei;

                    local_deriv = 2 * w * niei * grad.transpose();
                    local_hessian = 2 * (w * grad.transpose() * grad + w * niei * hess);
                    break;
                }

                case SFFGeneralType::kS2TanFormulation: {
                    assert(ndofs_per_edge == LibShell::MidedgeAngleGeneralTanFormulation::numExtraDOFs);
                    Eigen::Matrix<double, 1, 18 + 3 * LibShell::MidedgeAngleGeneralTanFormulation::numExtraDOFs> grad;
                    Eigen::Matrix<double, 18 + 3 * LibShell::MidedgeAngleGeneralTanFormulation::numExtraDOFs,
                                  18 + 3 * LibShell::MidedgeAngleGeneralTanFormulation::numExtraDOFs>
                        hess;
                    double niei = LibShell::MidedgeAngleGeneralTanFormulation::compute_niei(
                        mesh, cur_pos, edge_dofs, i, j, derivative ? &grad : nullptr, hessian ? &hess : nullptr);

                    energy += w * niei * niei;

                    local_deriv = 2 * w * niei * grad.transpose();
                    local_hessian = 2 * (w * grad.transpose() * grad + w * niei * hess);
                    break;
                }

                default: {
                    std::cerr << "Invalid type!" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            if (derivative) {
                for (int j = 0; j < 3; j++) {
                    derivative->segment<3>(3 * mesh.faceVertex(i, j)) += local_deriv.segment<3>(3 * j);
                    int oppidx = mesh.vertexOppositeFaceEdge(i, j);
                    if (oppidx != -1) {
                        derivative->segment<3>(3 * oppidx) += local_deriv.segment<3>(9 + 3 * j);
                    }

                    for (int k = 0; k < ndofs_per_edge; k++) {
                        (*derivative)[3 * cur_pos.rows() + ndofs_per_edge * mesh.faceEdge(i, j) + k] +=
                            local_deriv(18 + ndofs_per_edge * j + k);
                    }
                }
            }
            if (hessian) {
                if (is_proj) {
                    LibShell::projSymMatrix(local_hessian, LibShell::HessianProjectType::kMaxZero);
                }
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        for (int l = 0; l < 3; l++) {
                            for (int m = 0; m < 3; m++) {
                                hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                                          3 * mesh.faceVertex(i, k) + m,
                                                                          local_hessian(3 * j + l, 3 * k + m)));
                                int oppidxk = mesh.vertexOppositeFaceEdge(i, k);
                                if (oppidxk != -1)
                                    hessian->push_back(Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                                              3 * oppidxk + m,
                                                                              local_hessian(3 * j + l, 9 + 3 * k + m)));
                                int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                                if (oppidxj != -1)
                                    hessian->push_back(Eigen::Triplet<double>(3 * oppidxj + l,
                                                                              3 * mesh.faceVertex(i, k) + m,
                                                                              local_hessian(9 + 3 * j + l, 3 * k + m)));
                                if (oppidxj != -1 && oppidxk != -1)
                                    hessian->push_back(Eigen::Triplet<double>(
                                        3 * oppidxj + l, 3 * oppidxk + m, local_hessian(9 + 3 * j + l, 9 + 3 * k + m)));
                            }
                            for (int m = 0; m < ndofs_per_edge; m++) {
                                hessian->push_back(
                                    Eigen::Triplet<double>(3 * mesh.faceVertex(i, j) + l,
                                                           3 * nverts + ndofs_per_edge * mesh.faceEdge(i, k) + m,
                                                           local_hessian(3 * j + l, 18 + ndofs_per_edge * k + m)));
                                hessian->push_back(
                                    Eigen::Triplet<double>(3 * nverts + ndofs_per_edge * mesh.faceEdge(i, k) + m,
                                                           3 * mesh.faceVertex(i, j) + l,
                                                           local_hessian(18 + ndofs_per_edge * k + m, 3 * j + l)));
                                int oppidxj = mesh.vertexOppositeFaceEdge(i, j);
                                if (oppidxj != -1) {
                                    hessian->push_back(Eigen::Triplet<double>(
                                        3 * oppidxj + l, 3 * nverts + ndofs_per_edge * mesh.faceEdge(i, k) + m,
                                        local_hessian(9 + 3 * j + l, 18 + ndofs_per_edge * k + m)));
                                    hessian->push_back(Eigen::Triplet<double>(
                                        3 * nverts + ndofs_per_edge * mesh.faceEdge(i, k) + m, 3 * oppidxj + l,
                                        local_hessian(18 + ndofs_per_edge * k + m, 9 + 3 * j + l)));
                                }
                            }
                        }
                        for (int m = 0; m < ndofs_per_edge; m++) {
                            for (int n = 0; n < ndofs_per_edge; n++) {
                                hessian->push_back(Eigen::Triplet<double>(
                                    3 * nverts + ndofs_per_edge * mesh.faceEdge(i, j) + m,
                                    3 * nverts + ndofs_per_edge * mesh.faceEdge(i, k) + n,
                                    local_hessian(18 + ndofs_per_edge * j + m, 18 + ndofs_per_edge * k + n)));
                            }
                        }
                    }
                }
            }
        }
    }
    return energy;
}

void testEdgeVertorPerpTerm(const Eigen::MatrixXd& cur_pos,
                            const Eigen::VectorXd& edge_dofs,
                            const LibShell::MeshConnectivity& mesh,
                            const Eigen::VectorXd& edge_area,
                            SFFGeneralType general_type) {
    int nedges = mesh.nEdges();
    int nverts = cur_pos.rows();
    int numExtraDOFs = 2;
    if (general_type == SFFGeneralType::kGeneralFormulation) {
        numExtraDOFs = 4;
    }
    auto to_variables = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& cur_edge_dofs) {
        Eigen::VectorXd vars(3 * nverts + numExtraDOFs * nedges);
        vars.setZero();
        for (int i = 0; i < nverts; i++) {
            vars.segment<3>(3 * i) = pos.row(i);
        }
        for (int i = 0; i < nedges; i++) {
            vars.segment(3 * nverts + numExtraDOFs * i, numExtraDOFs) =
                cur_edge_dofs.segment(numExtraDOFs * i, numExtraDOFs);
        }
        return vars;
    };

    auto from_variable = [&](const Eigen::VectorXd& vars, Eigen::MatrixXd& pos, Eigen::VectorXd& cur_edge_dofs) {
        assert(vars.size() == 3 * nverts + numExtraDOFs * nedges);
        for (int i = 0; i < nverts; i++) {
            pos.row(i) = vars.segment<3>(3 * i);
        }
        for (int i = 0; i < nedges; i++) {
            cur_edge_dofs.segment(numExtraDOFs * i, numExtraDOFs) =
                vars.segment(3 * nverts + numExtraDOFs * i, numExtraDOFs);
        }
    };

    Eigen::VectorXd vars = to_variables(cur_pos, edge_dofs);
    Eigen::MatrixXd pos = cur_pos;
    Eigen::VectorXd cur_edge_dofs = edge_dofs;

    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hessian,
                    bool is_proj) {
        from_variable(x, pos, cur_edge_dofs);
        std::vector<Eigen::Triplet<double>> T;
        double val =
            edgeVectorPerpTerm(pos, cur_edge_dofs, mesh, edge_area, general_type, deriv, hessian ? &T : nullptr, false);

        if (hessian) {
            hessian->resize(3 * nverts + numExtraDOFs * nedges, 3 * nverts + numExtraDOFs * nedges);
            hessian->setFromTriplets(T.begin(), T.end());
        }
        return val;
    };

    OptSolver::TestFuncGradHessian(func, vars);
}

void optimizeEdgeDOFs(ShellEnergy& energy,
    const std::vector<Eigen::Matrix2d>& abars,
                      const Eigen::MatrixXd& cur_pos,
                      const LibShell::MeshConnectivity& mesh,
                      const Eigen::VectorXd& edge_area,
                      Eigen::VectorXd& edgeDOFs,
                      LibShell::ExtraEnergyTerms* extra_energy_terms = nullptr,
                      SFFGeneralType* general_type = nullptr) {
    double tol = 1e-5;
    int nposdofs = cur_pos.rows() * 3;
    int nedgedofs = edgeDOFs.size();

    std::vector<Eigen::Triplet<double>> Pcoeffs;
    std::vector<Eigen::Triplet<double>> Icoeffs;
    for (int i = 0; i < nedgedofs; i++) {
        Pcoeffs.push_back({i, nposdofs + i, 1.0});
        Icoeffs.push_back({i, i, 1.0});
    }
    Eigen::SparseMatrix<double> P(nedgedofs, nposdofs + nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());
    Eigen::SparseMatrix<double> I(nedgedofs, nedgedofs);
    I.setFromTriplets(Icoeffs.begin(), Icoeffs.end());

    Eigen::SparseMatrix<double> PT = P.transpose();

    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double total_energy = 0;

        double elastic_energy = energy.elasticEnergy(
            cur_pos, var, true, grad, hessian ? &hessian_triplets : nullptr,
            psd_proj ? LibShell::HessianProjectType::kMaxZero : LibShell::HessianProjectType::kNone);

        total_energy += elastic_energy;

        if (grad) {
            *grad = P * (*grad);
        }

        if (hessian) {
            hessian->resize(var.size() + 3 * cur_pos.rows(), var.size() + 3 * cur_pos.rows());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            *hessian = P * (*hessian) * PT;
        }

        if (extra_energy_terms || edgeDOFs.size() == 4 * edge_area.size()) {
            Eigen::VectorXd mag_comp_deriv, direct_perp_deriv, mag_sq_change_deriv;
            std::vector<Eigen::Triplet<double>> mag_comp_triplets, direct_perp_triplets, mag_sq_change_triplets;
            double mag_comp = extra_energy_terms->compute_magnitude_compression_energy(
                var, mesh, grad ? &mag_comp_deriv : nullptr, hessian ? &mag_comp_triplets : nullptr, psd_proj);
            double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
                cur_pos, var, mesh, abars, grad ? &direct_perp_deriv : nullptr, hessian ? &direct_perp_triplets : nullptr,
                psd_proj);
            double mag_sq_change = extra_energy_terms->compute_magnitude_sq_change_energy(
                var, mesh, abars, grad ? &mag_sq_change_deriv : nullptr, hessian ? &mag_sq_change_triplets : nullptr,
                psd_proj);

            total_energy += mag_comp + mag_sq_change;
            total_energy += direct_perp;

            if (grad) {
                *grad += (mag_comp_deriv + mag_sq_change_deriv);
                *grad += P * direct_perp_deriv;
            }
            if (hessian) {
                Eigen::SparseMatrix<double> mag_comp_hess, direct_perp_hess, mag_sq_change_hess;

                mag_comp_hess.resize(var.size(), var.size());
                mag_comp_hess.setFromTriplets(mag_comp_triplets.begin(), mag_comp_triplets.end());

                direct_perp_hess.resize(var.size() + 3 * cur_pos.rows(), var.size() + 3 * cur_pos.rows());
                direct_perp_hess.setFromTriplets(direct_perp_triplets.begin(), direct_perp_triplets.end());
                direct_perp_hess = P * direct_perp_hess * PT;

                mag_sq_change_hess.resize(var.size(), var.size());
                mag_sq_change_hess.setFromTriplets(mag_sq_change_triplets.begin(), mag_sq_change_triplets.end());

                *hessian += mag_comp_hess + mag_sq_change_hess;

                *hessian += direct_perp_hess;
            }
        }

        if (general_type) {
            Eigen::VectorXd direct_perp_deriv;
            std::vector<Eigen::Triplet<double>> direct_perp_triplets;

            double direct_perp =
                edgeVectorPerpTerm(cur_pos, var, mesh, edge_area, *general_type, grad ? &direct_perp_deriv : nullptr,
                                   hessian ? &direct_perp_triplets : nullptr, psd_proj);

            total_energy += direct_perp;

            if (grad) {
                *grad += P * direct_perp_deriv;
            }
            if (hessian) {
                Eigen::SparseMatrix<double> direct_perp_hess;

                direct_perp_hess.resize(var.size() + 3 * cur_pos.rows(), var.size() + 3 * cur_pos.rows());
                direct_perp_hess.setFromTriplets(direct_perp_triplets.begin(), direct_perp_triplets.end());
                direct_perp_hess = P * direct_perp_hess * PT;

                *hessian += direct_perp_hess;
            }
        }

        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "------------------------ At beginning -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, edgeDOFs, true, NULL, NULL)
              << std::endl;

    if(extra_energy_terms && edgeDOFs.size() == 4 * mesh.nEdges()) {
        double mag_comp = extra_energy_terms->compute_magnitude_compression_energy(
                edgeDOFs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
            cur_pos, edgeDOFs, mesh, abars, nullptr, nullptr,
            false);
        double mag_sq_change = extra_energy_terms->compute_magnitude_sq_change_energy(
            edgeDOFs, mesh, abars, nullptr, nullptr, false);

        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "m^2 ||dm||^2: " << mag_sq_change << std::endl;
    }

    if(general_type) {
        double direct_perp =
               edgeVectorPerpTerm(cur_pos, edgeDOFs, mesh, edge_area, *general_type, nullptr,
                                  nullptr, false);
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
    }

    OptSolver::NewtonSolver(obj_func, find_max_step, edgeDOFs, 1000, 1e-5, 1e-15, 1e-15, true, true, true);

    std::cout << "------------------------ At end -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, edgeDOFs, true, NULL, NULL)
              << std::endl;

    if(extra_energy_terms && edgeDOFs.size() == 4 * mesh.nEdges()) {
        double mag_comp = extra_energy_terms->compute_magnitude_compression_energy(
                edgeDOFs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
            cur_pos, edgeDOFs, mesh, abars, nullptr, nullptr,
            false);
        double mag_sq_change = extra_energy_terms->compute_magnitude_sq_change_energy(
            edgeDOFs, mesh, abars, nullptr, nullptr, false);

        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "m^2 ||dm||^2: " << mag_sq_change << std::endl;
    }

    if(general_type) {
        double direct_perp =
               edgeVectorPerpTerm(cur_pos, edgeDOFs, mesh, edge_area, *general_type, nullptr,
                                  nullptr, false);
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
    }
}

Energies measureCylinderEnergy(const LibShell::MeshConnectivity& mesh,
                               const Eigen::MatrixXd& rest_pos,
                               const Eigen::MatrixXd& cur_pos,
                               double thickness,
                               double lame_alpha,
                               double lame_beta,
                               double cur_radius,
                               double cur_height,
                               Eigen::MatrixXd& nhForces,
                               Eigen::MatrixXd& qbForces) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edge_dofs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edge_dofs, mesh, rest_pos);

    Eigen::VectorXd zero_s1_dir_edge_dofs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zero_s1_dir_edge_dofs, mesh, rest_pos);

    Eigen::VectorXd half_pi_zero_s2_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(half_pi_zero_s2_dir_edge_dofs, mesh, rest_pos);

    Eigen::VectorXd general_unit_half_pi_zero_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralFormulation::initializeExtraDOFs(general_unit_half_pi_zero_dir_edge_dofs, mesh,
                                                                  rest_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state, s1_dir_rest_state, s2_dir_rest_state, general_dir_rest_state;

    // set uniform thicknesses
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    s1_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s1_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s1_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    s2_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s2_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s2_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    general_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    general_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    general_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                       rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, rest_pos, edge_dofs,
                                                                                        rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                        s1_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::secondFundamentalForms(
        mesh, rest_pos, zero_s1_dir_edge_dofs, s1_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                               s2_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::secondFundamentalForms(
        mesh, rest_pos, half_pi_zero_s2_dir_edge_dofs, s2_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::firstFundamentalForms(
        mesh, rest_pos, general_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::secondFundamentalForms(
        mesh, rest_pos, general_unit_half_pi_zero_dir_edge_dofs, general_dir_rest_state.bbars);

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, cur_pos, edge_dofs,
                                                                                        cur_bs);

    Eigen::VectorXd rest_edge_dofs = edge_dofs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, rest_pos, rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);
    result.stvk = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);

    StVKS1DirectorSinShellEnergy stvk_s1_dir_sin_energy_model(mesh, s1_dir_rest_state);
    StVKS1DirectorTanShellEnergy stvk_s1_dir_tan_energy_model(mesh, s1_dir_rest_state);
    StVKS2DirectorSinShellEnergy stvk_s2_dir_sin_energy_model(mesh, s2_dir_rest_state);
    StVKS2DirectorTanShellEnergy stvk_s2_dir_tan_energy_model(mesh, s2_dir_rest_state);
    StVKGeneralDirectorSinShellEnergy stvk_general_dir_energy_model(mesh, general_dir_rest_state);

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    Eigen::VectorXd s1_dir_sin_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
    // optimizeEdgeDOFs(stvk_s1_dir_sin_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_sin_edge_dofs);
    result.stvk_s1_dir_sin =
        stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s1_dir_tan_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
    // optimizeEdgeDOFs(stvk_s1_dir_tan_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_tan_edge_dofs);
    result.stvk_s1_dir_tan =
        stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd general_dir_edge_dofs = general_unit_half_pi_zero_dir_edge_dofs;
    SFFGeneralType general_type = SFFGeneralType::kGeneralFormulation;
    std::cout << "============= Optimizing edge direction (General dir) =========== " << std::endl;
    LibShell::ExtraEnergyTerms extra_energy_terms;
    extra_energy_terms.initialization(rest_pos, mesh, youngs, shear, thickness, quad_order);
    optimizeEdgeDOFs(stvk_general_dir_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, general_dir_edge_dofs,
                     &extra_energy_terms, nullptr);
    // &general_type);

    double mag_comp =
        extra_energy_terms.compute_magnitude_compression_energy(general_dir_edge_dofs, mesh, nullptr, nullptr, false);
    double dir_perp = extra_energy_terms.compute_vector_perp_tangent_energy(cur_pos, general_dir_edge_dofs, mesh, s2_dir_rest_state.abars,
                                                                           nullptr, nullptr, false);
    // edgeVectorPerpTerm(cur_pos, general_dir_edge_dofs, mesh, edge_area, general_type, nullptr, nullptr, false);
    double mag_sq_change =
        extra_energy_terms.compute_magnitude_sq_change_energy(general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);

    result.stvk_general_dir =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_dir_edge_dofs, true, nullptr, nullptr) + mag_comp +
        dir_perp + mag_sq_change;

    std::cout << "total energy: " << result.stvk_general_dir << std::endl;

    Eigen::VectorXd s2_dir_sin_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_sin_general_type = SFFGeneralType::kS2SinFormulation;
    std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
    // optimizeEdgeDOFs(stvk_s2_dir_sin_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, s2_dir_sin_edge_dofs, nullptr,
    //                 nullptr);
                     // &s2_sin_general_type);
    // double s2_sin_dir_perp = edgeVectorPerpTerm(cur_pos, s2_dir_sin_edge_dofs, mesh, edge_area, s2_sin_general_type,
    //                                             nullptr, nullptr, false);
    result.stvk_s2_dir_sin = 0;
        // stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, true, nullptr, nullptr);
    // +s2_sin_dir_perp;

    Eigen::VectorXd s2_dir_tan_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_tan_general_type = SFFGeneralType::kS2TanFormulation;
    std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
    // optimizeEdgeDOFs(stvk_s2_dir_tan_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, s2_dir_tan_edge_dofs, nullptr,
    //                 nullptr);
                     // &s2_tan_general_type);
    // double s2_tan_dir_perp = edgeVectorPerpTerm(cur_pos, s2_dir_tan_edge_dofs, mesh, edge_area, s2_tan_general_type,
                                                // nullptr, nullptr, false);
    result.stvk_s2_dir_tan = 0;
        // stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, true, nullptr, nullptr);
    // + s2_tan_dir_perp;

    // ground truth energy
    // W = PI * r
    // r(x,y) = (r cos[x/r], r sin[x/r], y)^T
    // dr(x,y) = ((-sin[x/r], 0),
    //            ( cos[x/r], 0),
    //            ( 0, 1 ))
    Eigen::Matrix2d abar;
    abar.setIdentity();

    // n = (-sin[x/r], cos[x/r], 0) x (0, 0, 1) = ( cos[x/r], sin[x/r], 0 )
    // dn = ((-sin[x/r]/r, 0),
    //       ( cos[x/r]/r, 0),
    //       ( 0, 0 ))
    // b = dr^T dn = ((1/r, 0), (0, 0))
    Eigen::Matrix2d b;
    b << 1.0 / cur_radius, 0, 0, 0;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm = lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = PI * cur_radius * cur_height;

    result.exact = svnorm * coeff * area;

    // std::vector<Eigen::Vector3d> tan_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_tan_edge_dofs, s2_tan_general_type);
    // std::vector<Eigen::Vector3d> sin_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_sin_edge_dofs, s2_sin_general_type);
    // std::vector<Eigen::Vector3d> general_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, general_dir_edge_dofs, general_type);
    // std::vector<Eigen::Vector3d> mid_surface_face_edge_normals = get_midsurf_face_edge_normal_vectors(cur_pos, mesh);
    // pt_mesh->addVectorQuantity("Tan Edge Normals", tan_face_edge_normals);
    // pt_mesh->addVectorQuantity("Sin Edge Normals", sin_face_edge_normals);
    // pt_mesh->addVectorQuantity("General Edge Normals", general_face_edge_normals);
    // pt_mesh->addVectorQuantity("Mid Surface Edge Normals", mid_surface_face_edge_normals);
    //
    // std::vector<double> mag_comp_scalars, perp_scalars, mag_sq_change_scalars;
    // for(int i = 0; i < mesh.nFaces(); i++) {
    //     mag_comp_scalars.push_back(extra_energy_terms.compute_magnitude_compression_energy_perface(
    //         general_dir_edge_dofs, mesh, i, nullptr, nullptr, false));
    //     perp_scalars.push_back(extra_energy_terms.compute_vector_perp_tangent_energy_perface(cur_pos, general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
    //     mag_sq_change_scalars.push_back(extra_energy_terms.compute_magnitude_sq_change_energy_perface(general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
    // }
    //
    // surface_mesh->addFaceScalarQuantity("mag compression", mag_comp_scalars);
    // auto perp_plot = surface_mesh->addFaceScalarQuantity("perp", perp_scalars);
    // auto [min_val, max_val] = std::minmax_element(perp_scalars.begin(), perp_scalars.end());
    // perp_plot->setMapRange({*min_val, *max_val});
    // surface_mesh->addFaceScalarQuantity("mag sq change", mag_sq_change_scalars);

    return result;
}

Energies measureSphereEnergy(const LibShell::MeshConnectivity& mesh,
                             const Eigen::MatrixXd& cur_pos,
                             double thickness,
                             double lame_alpha,
                             double lame_beta,
                             double radius) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edge_dofs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edge_dofs, mesh, cur_pos);

    Eigen::VectorXd zero_s1_dir_edge_dofs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zero_s1_dir_edge_dofs, mesh, cur_pos);

    Eigen::VectorXd half_pi_zero_s2_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(half_pi_zero_s2_dir_edge_dofs, mesh, cur_pos);

    Eigen::VectorXd general_unit_half_pi_zero_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralFormulation::initializeExtraDOFs(general_unit_half_pi_zero_dir_edge_dofs, mesh,
                                                                  cur_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state, s1_dir_rest_state, s2_dir_rest_state, general_dir_rest_state;

    // set uniform thicknesses
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    s1_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s1_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s1_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    s2_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s2_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s2_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    general_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    general_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    general_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, cur_pos, rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, cur_pos, edge_dofs,
                                                                                        rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::firstFundamentalForms(mesh, cur_pos,
                                                                                        s1_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::secondFundamentalForms(
        mesh, cur_pos, zero_s1_dir_edge_dofs, s1_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::firstFundamentalForms(mesh, cur_pos,
                                                                                               s2_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::secondFundamentalForms(
        mesh, cur_pos, half_pi_zero_s2_dir_edge_dofs, s2_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::firstFundamentalForms(
        mesh, cur_pos, general_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::secondFundamentalForms(
        mesh, cur_pos, general_unit_half_pi_zero_dir_edge_dofs, general_dir_rest_state.bbars);

    for (int i = 0; i < mesh.nFaces(); i++) {
        rest_state.bbars[i].setZero();
        s1_dir_rest_state.bbars[i].setZero();
        s2_dir_rest_state.bbars[i].setZero();
        general_dir_rest_state.bbars[i].setZero();
    }

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, cur_pos, edge_dofs,
                                                                                        cur_bs);

    Eigen::VectorXd rest_edge_dofs = edge_dofs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, cur_pos, rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);
    result.stvk = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);

    StVKS1DirectorSinShellEnergy stvk_s1_dir_sin_energy_model(mesh, s1_dir_rest_state);
    StVKS1DirectorTanShellEnergy stvk_s1_dir_tan_energy_model(mesh, s1_dir_rest_state);
    StVKS2DirectorSinShellEnergy stvk_s2_dir_sin_energy_model(mesh, s2_dir_rest_state);
    StVKS2DirectorTanShellEnergy stvk_s2_dir_tan_energy_model(mesh, s2_dir_rest_state);
    StVKGeneralDirectorSinShellEnergy stvk_general_dir_energy_model(mesh, general_dir_rest_state);

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    Eigen::VectorXd s1_dir_sin_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_sin_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_sin_edge_dofs);
    result.stvk_s1_dir_sin =
        stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s1_dir_tan_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_tan_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_tan_edge_dofs);
    result.stvk_s1_dir_tan =
        stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd general_dir_edge_dofs = general_unit_half_pi_zero_dir_edge_dofs;
    SFFGeneralType general_type = SFFGeneralType::kGeneralFormulation;
    std::cout << "============= Optimizing edge direction (General dir) =========== " << std::endl;
    LibShell::ExtraEnergyTerms extra_energy_terms;
    extra_energy_terms.initialization(cur_pos, mesh, youngs, shear, thickness, quad_order);
    optimizeEdgeDOFs(stvk_general_dir_energy_model, general_dir_rest_state.abars, cur_pos, mesh, edge_area, general_dir_edge_dofs,
                     &extra_energy_terms, &general_type);

    double mag_comp =
        extra_energy_terms.compute_magnitude_compression_energy(general_dir_edge_dofs, mesh, nullptr, nullptr, false);
    double dir_perp =
        edgeVectorPerpTerm(cur_pos, general_dir_edge_dofs, mesh, edge_area, general_type, nullptr, nullptr, false);
    double mag_sq_change =
        extra_energy_terms.compute_magnitude_sq_change_energy(general_dir_edge_dofs, mesh, general_dir_rest_state.abars, nullptr, nullptr, false);

    result.stvk_general_dir =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_dir_edge_dofs, true, nullptr, nullptr) + mag_comp +
        dir_perp + mag_sq_change;

    Eigen::VectorXd s2_dir_sin_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_sin_general_type = SFFGeneralType::kS2SinFormulation;
    std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_sin_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, s2_dir_sin_edge_dofs, nullptr,
                     &s2_sin_general_type);
    double s2_sin_dir_perp = edgeVectorPerpTerm(cur_pos, s2_dir_sin_edge_dofs, mesh, edge_area, s2_sin_general_type,
                                                nullptr, nullptr, false);
    result.stvk_s2_dir_sin =
        stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, true, nullptr, nullptr) +
        s2_sin_dir_perp;

    Eigen::VectorXd s2_dir_tan_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_tan_general_type = SFFGeneralType::kS2TanFormulation;
    std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_tan_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, s2_dir_tan_edge_dofs, nullptr,
                     &s2_tan_general_type);
    double s2_tan_dir_perp = edgeVectorPerpTerm(cur_pos, s2_dir_tan_edge_dofs, mesh, edge_area, s2_tan_general_type,
                                                nullptr, nullptr, false);
    result.stvk_s2_dir_tan =
        stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, true, nullptr, nullptr) +
        s2_tan_dir_perp;

    // ground truth energy
    Eigen::Matrix2d abar;
    abar.setIdentity();

    Eigen::Matrix2d b;
    b << 1.0 / radius, 0, 0, 1.0 / radius;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm = lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = 4.0 * PI * radius * radius;

    result.exact = svnorm * coeff * area;

    std::vector<Eigen::Vector3d> tan_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_tan_edge_dofs, s2_tan_general_type);
    std::vector<Eigen::Vector3d> sin_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_sin_edge_dofs, s2_sin_general_type);
    std::vector<Eigen::Vector3d> general_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, general_dir_edge_dofs, general_type);
    std::vector<Eigen::Vector3d> mid_surface_face_edge_normals = get_midsurf_face_edge_normal_vectors(cur_pos, mesh);
    pt_mesh->addVectorQuantity("Tan Edge Normals", tan_face_edge_normals);
    pt_mesh->addVectorQuantity("Sin Edge Normals", sin_face_edge_normals);
    pt_mesh->addVectorQuantity("General Edge Normals", general_face_edge_normals);
    pt_mesh->addVectorQuantity("Mid Surface Edge Normals", mid_surface_face_edge_normals);

    std::vector<double> mag_comp_scalars, perp_scalars, mag_sq_change_scalars;
    for(int i = 0; i < mesh.nFaces(); i++) {
        mag_comp_scalars.push_back(extra_energy_terms.compute_magnitude_compression_energy_perface(
            general_dir_edge_dofs, mesh, i, nullptr, nullptr, false));
        perp_scalars.push_back(extra_energy_terms.compute_vector_perp_tangent_energy_perface(cur_pos, general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
        mag_sq_change_scalars.push_back(extra_energy_terms.compute_magnitude_sq_change_energy_perface(general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
    }

    surface_mesh->addFaceScalarQuantity("mag compression", mag_comp_scalars);
    auto perp_plot = surface_mesh->addFaceScalarQuantity("perp", perp_scalars);
    auto [min_val, max_val] = std::minmax_element(perp_scalars.begin(), perp_scalars.end());
    perp_plot->setMapRange({*min_val, *max_val});
    surface_mesh->addFaceScalarQuantity("mag sq change", mag_sq_change_scalars);

    return result;
}

Energies measureFoldEnergy(const LibShell::MeshConnectivity& mesh,
                           const Eigen::MatrixXd& rest_pos,
                           const Eigen::MatrixXd& cur_pos,
                           double thickness,
                           double lame_alpha,
                           double lame_beta) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edge_dofs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edge_dofs, mesh, rest_pos);

    Eigen::VectorXd zero_s1_dir_edge_dofs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zero_s1_dir_edge_dofs, mesh, rest_pos);

    Eigen::VectorXd half_pi_zero_s2_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(half_pi_zero_s2_dir_edge_dofs, mesh, rest_pos);

    Eigen::VectorXd general_unit_half_pi_zero_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralFormulation::initializeExtraDOFs(general_unit_half_pi_zero_dir_edge_dofs, mesh,
                                                                  rest_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state, s1_dir_rest_state, s2_dir_rest_state, general_dir_rest_state;

    // set uniform thicknesses
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    s1_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s1_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s1_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    s2_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s2_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s2_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    general_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    general_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    general_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                       rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, rest_pos, edge_dofs,
                                                                                        rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                        s1_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::secondFundamentalForms(
        mesh, rest_pos, zero_s1_dir_edge_dofs, s1_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                               s2_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::secondFundamentalForms(
        mesh, rest_pos, half_pi_zero_s2_dir_edge_dofs, s2_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::firstFundamentalForms(
        mesh, rest_pos, general_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::secondFundamentalForms(
        mesh, rest_pos, general_unit_half_pi_zero_dir_edge_dofs, general_dir_rest_state.bbars);

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, cur_pos, edge_dofs,
                                                                                        cur_bs);

    Eigen::VectorXd rest_edge_dofs = edge_dofs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, rest_pos, rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);
    result.stvk = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);

    StVKS1DirectorSinShellEnergy stvk_s1_dir_sin_energy_model(mesh, s1_dir_rest_state);
    StVKS1DirectorTanShellEnergy stvk_s1_dir_tan_energy_model(mesh, s1_dir_rest_state);
    StVKS2DirectorSinShellEnergy stvk_s2_dir_sin_energy_model(mesh, s2_dir_rest_state);
    StVKS2DirectorTanShellEnergy stvk_s2_dir_tan_energy_model(mesh, s2_dir_rest_state);
    StVKGeneralDirectorSinShellEnergy stvk_general_dir_energy_model(mesh, general_dir_rest_state);

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    Eigen::VectorXd s1_dir_sin_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_sin_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_sin_edge_dofs);
    result.stvk_s1_dir_sin =
        stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s1_dir_tan_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_tan_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_tan_edge_dofs);
    result.stvk_s1_dir_tan =
        stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s2_dir_sin_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_sin_general_type = SFFGeneralType::kS2SinFormulation;
    std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_sin_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, s2_dir_sin_edge_dofs, nullptr,
                     &s2_sin_general_type);
    double s2_sin_dir_perp = edgeVectorPerpTerm(cur_pos, s2_dir_sin_edge_dofs, mesh, edge_area, s2_sin_general_type,
                                                nullptr, nullptr, false);
    result.stvk_s2_dir_sin =
        stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, true, nullptr, nullptr) +
        s2_sin_dir_perp;

    Eigen::VectorXd s2_dir_tan_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_tan_general_type = SFFGeneralType::kS2TanFormulation;
    std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_tan_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area, s2_dir_tan_edge_dofs, nullptr,
                     &s2_tan_general_type);
    double s2_tan_dir_perp = edgeVectorPerpTerm(cur_pos, s2_dir_tan_edge_dofs, mesh, edge_area, s2_tan_general_type,
                                                nullptr, nullptr, false);
    result.stvk_s2_dir_tan =
        stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, true, nullptr, nullptr) +
        s2_tan_dir_perp;

    Eigen::VectorXd general_dir_edge_dofs = general_unit_half_pi_zero_dir_edge_dofs;
    SFFGeneralType general_type = SFFGeneralType::kGeneralFormulation;
    std::cout << "============= Optimizing edge direction (General dir) =========== " << std::endl;
    LibShell::ExtraEnergyTerms extra_energy_terms;
    extra_energy_terms.initialization(rest_pos, mesh, youngs, shear, thickness, quad_order);
    optimizeEdgeDOFs(stvk_general_dir_energy_model, general_dir_rest_state.abars, cur_pos, mesh, edge_area, general_dir_edge_dofs,
                     &extra_energy_terms, &general_type);

    double mag_comp =
        extra_energy_terms.compute_magnitude_compression_energy(general_dir_edge_dofs, mesh, nullptr, nullptr, false);
    double dir_perp =
        edgeVectorPerpTerm(cur_pos, general_dir_edge_dofs, mesh, edge_area, general_type, nullptr, nullptr, false);
    double mag_sq_change =
        extra_energy_terms.compute_magnitude_sq_change_energy(general_dir_edge_dofs, mesh, general_dir_rest_state.abars, nullptr, nullptr, false);

    result.stvk_general_dir =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_dir_edge_dofs, true, nullptr, nullptr) + mag_comp +
        dir_perp + mag_sq_change;

    // visualization
    std::vector<Eigen::Vector3d> tan_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_tan_edge_dofs, s2_tan_general_type);
    std::vector<Eigen::Vector3d> sin_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_sin_edge_dofs, s2_sin_general_type);
    std::vector<Eigen::Vector3d> general_face_edge_normals = get_face_edge_normal_vectors(cur_pos, mesh, general_dir_edge_dofs, general_type);
    std::vector<Eigen::Vector3d> mid_surface_face_edge_normals = get_midsurf_face_edge_normal_vectors(cur_pos, mesh);
    pt_mesh->addVectorQuantity("Tan Edge Normals", tan_face_edge_normals);
    pt_mesh->addVectorQuantity("Sin Edge Normals", sin_face_edge_normals);
    pt_mesh->addVectorQuantity("General Edge Normals", general_face_edge_normals);
    pt_mesh->addVectorQuantity("Mid Surface Edge Normals", mid_surface_face_edge_normals);

    std::vector<double> mag_comp_scalars, perp_scalars, mag_sq_change_scalars;
    for(int i = 0; i < mesh.nFaces(); i++) {
        mag_comp_scalars.push_back(extra_energy_terms.compute_magnitude_compression_energy_perface(
            general_dir_edge_dofs, mesh, i, nullptr, nullptr, false));
        perp_scalars.push_back(extra_energy_terms.compute_vector_perp_tangent_energy_perface(cur_pos, general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
        mag_sq_change_scalars.push_back(extra_energy_terms.compute_magnitude_sq_change_energy_perface(general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
    }

    surface_mesh->addFaceScalarQuantity("mag compression", mag_comp_scalars);
    surface_mesh->addFaceScalarQuantity("perp", perp_scalars);
    surface_mesh->addFaceScalarQuantity("mag sq change", mag_sq_change_scalars);

    result.exact = 0;

    return result;
}

static void gererated_foleded_mesh(
    int N, int M, double fold_theta, Eigen::MatrixXd& rest_V, Eigen::MatrixXi& rest_F, Eigen::MatrixXd& fold_V) {
    double constexpr R = 1;
    double constexpr H = 5;

    std::vector<Eigen::Vector3d> rest_pos, fold_pos;
    std::vector<Eigen::Vector3i> rest_faces;

    for (int i = 0; i <= 2 * N; i++) {
        for (int j = 0; j <= M; j++) {
            double x = (i - N) * M_PI * R / N;
            double y = j * H / M;
            rest_pos.push_back(Eigen::Vector3d{x, y, 0});

            if (i <= N) {
                fold_pos.push_back(Eigen::Vector3d(x, y, 0));
            } else {
                fold_pos.push_back(Eigen::Vector3d(x * std::cos(fold_theta), y, x * std::sin(fold_theta)));
            }

            // T theta = i * M_PI / N;
            // cylinder_pos.push_back(Vector<T, dim>{-R * std::sin(theta), y, R + R * std::cos(theta)});
        }
    }

    for (int i = 0; i < 2 * N; i++) {
        for (int j = 0; j < M; j++) {
            int k = i * (M + 1) + j;
            rest_faces.push_back(Eigen::Vector3i{k, k + 1, k + M + 1});
            rest_faces.push_back(Eigen::Vector3i{k + 1, k + M + 2, k + M + 1});
        }
    }

    rest_V.resize(rest_pos.size(), 3);
    for (int vid = 0; vid < rest_pos.size(); vid++) {
        rest_V.row(vid) = rest_pos[vid].transpose();
    }

    fold_V.resize(fold_pos.size(), 3);
    for (int vid = 0; vid < fold_pos.size(); vid++) {
        fold_V.row(vid) = fold_pos[vid].transpose();
    }

    rest_F.resize(rest_faces.size(), 3);
    for (int fid = 0; fid < rest_faces.size(); fid++) {
        rest_F.row(fid) = rest_faces[fid].transpose();
    }
}

int main(int argc, char* argv[]) {
    // cokeRadius = 0.0325;
    // cokeHeight = 0.122;
    cokeRadius = 1;
    cokeHeight = 1;
    sphereRadius = 0.05;

    // triangleArea = 0.0000001;
    triangleArea = 0.00001;

    // curMeshType = MeshType::MT_CYLINDER_REGULAR;
    curMeshType = MeshType::MT_CYLINDER_IRREGULAR;
    // curMeshType = MeshType::MT_SPHERE;
    // curMeshType = MeshType::MT_FOLD_MESH;

    Energies curenergies;
    QBEnergy = 0;

    Eigen::MatrixXd nhForces;
    Eigen::MatrixXd qbForces;

    // set up material parameters
    youngs = 1.0;
    thickness = 1.0;  // 0.00010;
    poisson = 0;
    shear = youngs / (2.0 * (1.0 + poisson));

    // quad order
    quad_order = 2;

    Eigen::MatrixXd origV;
    Eigen::MatrixXd rolledV;
    Eigen::MatrixXi F;

    double lame_alpha, lame_beta;
    lameParameters(lame_alpha, lame_beta);
    double cur_radius = cokeRadius;
    double cur_height = cokeHeight;

    int steps = 5;
    double multiplier = 4;
    std::ofstream log("log.txt");
    // Assuming log is an output stream
    log << std::left;  // Left-align columns
    log << std::setw(15) << "#V"
        << "| " << std::setw(15) << "exact energy"
        << "| " << std::setw(25) << "Quadratic bending energy"
        << "| " << std::setw(20) << "StVK energy"
        << "| " << std::setw(20) << "StVK_S1_sin energy"
        << "| " << std::setw(20) << "StVK_S1_tan energy"
        << "| " << std::setw(20) << "StVK_S2_sin energy"
        << "| " << std::setw(20) << "StVK_S2_tan energy"
        << "| " << std::setw(20) << "StVK_General energy" << std::endl;
    if (curMeshType == MeshType::MT_SPHERE) {
        makeSphere(sphereRadius, triangleArea, origV, F);
        LibShell::MeshConnectivity mesh(F);
        rolledV = origV;

        polyscope::init();

        surface_mesh = polyscope::registerSurfaceMesh("Current mesh", rolledV, F);
        std::vector<Eigen::Vector3d> face_edge_midpts = {};
        for (int i = 0; i < mesh.nFaces(); i++) {
            for (int j = 0; j < 3; j++) {
                int eid = mesh.faceEdge(i, j);
                Eigen::Vector3d midpt =
                    (rolledV.row(mesh.edgeVertex(eid, 0)) + rolledV.row(mesh.edgeVertex(eid, 1))) / 2.0;
                face_edge_midpts.push_back(midpt);
            }
        }
        pt_mesh = polyscope::registerPointCloud("Face edge midpoints", face_edge_midpts);

        polyscope::state::userCallback = [&]() {
            if (ImGui::Button("Measure Fold Energy", ImVec2(-1, 0))) {
                curenergies = measureSphereEnergy(mesh, origV, thickness, lame_alpha, lame_beta, sphereRadius);
                log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| " << std::setw(25)
                    << curenergies.quadratic_bending << "| " << std::setw(20) << curenergies.stvk << "| " << std::setw(20)
                    << curenergies.stvk_s1_dir_sin << "| " << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
                    << std::setw(20) << curenergies.stvk_s2_dir_sin << "| " << std::setw(20) << curenergies.stvk_s2_dir_tan
                    << "| " << std::setw(20) << curenergies.stvk_general_dir << std::endl;
            }
        };
        // for (int step = 0; step < steps; step++) {
        //     std::stringstream ss;
        //     ss << "sphere_ " << step << ".ply";
        //     igl::writePLY(ss.str(), origV, F);
        //
        //     curenergies = measureSphereEnergy(mesh, origV, thickness, lame_alpha, lame_beta, sphereRadius);
        //     log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| " << std::setw(25)
        //         << curenergies.quadratic_bending << "| " << std::setw(20) << curenergies.stvk << "| " << std::setw(20)
        //         << curenergies.stvk_s1_dir_sin << "| " << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
        //         << std::setw(20) << curenergies.stvk_s2_dir_sin << "| " << std::setw(20) << curenergies.stvk_s2_dir_tan
        //         << "| " << std::setw(20) << curenergies.stvk_general_dir << std::endl;
        //     triangleArea *= multiplier;
        //     makeSphere(sphereRadius, triangleArea, origV, F);
        //     mesh = LibShell::MeshConnectivity(F);
        // }
    } else if (curMeshType == MeshType::MT_CYLINDER_IRREGULAR || curMeshType == MeshType::MT_CYLINDER_REGULAR) {
        makeCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV,
                 rolledV, F, 6.0 / 4 * M_PI);
        LibShell::MeshConnectivity mesh(F);

        // polyscope::init();
        //
        // surface_mesh = polyscope::registerSurfaceMesh("Current mesh", rolledV, F);
        // std::vector<Eigen::Vector3d> face_edge_midpts = {};
        // for (int i = 0; i < mesh.nFaces(); i++) {
        //     for (int j = 0; j < 3; j++) {
        //         int eid = mesh.faceEdge(i, j);
        //         Eigen::Vector3d midpt =
        //             (rolledV.row(mesh.edgeVertex(eid, 0)) + rolledV.row(mesh.edgeVertex(eid, 1))) / 2.0;
        //         face_edge_midpts.push_back(midpt);
        //     }
        // }
        // pt_mesh = polyscope::registerPointCloud("Face edge midpoints", face_edge_midpts);
        //
        // polyscope::state::userCallback = [&]() {
        //     if (ImGui::Button("Measure Fold Energy", ImVec2(-1, 0))) {
        //         curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, lame_alpha, lame_beta, cur_radius,
        //                                         cur_height, nhForces, qbForces);
        //
        //         log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| " << std::setw(25)
        //             << curenergies.quadratic_bending << "| " << std::setw(20) << curenergies.stvk << "| " << std::setw(20)
        //             << curenergies.stvk_s1_dir_sin << "| " << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
        //             << std::setw(20) << curenergies.stvk_s2_dir_sin << "| " << std::setw(20) << curenergies.stvk_s2_dir_tan
        //             << "| " << std::setw(20) << curenergies.stvk_general_dir << std::endl;
        //     }
        // };
        //
        // polyscope::show();

        for (int step = 0; step < steps; step++) {
            curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, lame_alpha, lame_beta, cur_radius,
                                                cur_height, nhForces, qbForces);

            log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| " << std::setw(25)
                << curenergies.quadratic_bending << "| " << std::setw(20) << curenergies.stvk << "| " << std::setw(20)
                << curenergies.stvk_s1_dir_sin << "| " << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
                << std::setw(20) << curenergies.stvk_s2_dir_sin << "| " << std::setw(20) << curenergies.stvk_s2_dir_tan
                << "| " << std::setw(20) << curenergies.stvk_general_dir << std::endl;
            triangleArea *= multiplier;
            makeHalfCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV,
                             rolledV, F);
            mesh = LibShell::MeshConnectivity(F);
        }
    } else {
        Eigen::MatrixXd fold_V;
        gererated_foleded_mesh(1, 1, M_PI * 0.9, origV, F, fold_V);
        LibShell::MeshConnectivity mesh(F);
        Eigen::VectorXd test_edge_dofs;
        LibShell::MidedgeAngleGeneralFormulation::initializeExtraDOFs(test_edge_dofs, mesh, origV);
        LibShell::ExtraEnergyTerms extra_terms;
        std::vector<Eigen::Matrix2d> abars;
        LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, origV,
                                                                                       abars);
        extra_terms.initialization(origV, mesh, 1, 2, 1, 2);
        extra_terms.test_compute_magnitude_compression_energy(mesh, test_edge_dofs);
        extra_terms.test_compute_vector_perp_tangent_energy(mesh, abars, fold_V, test_edge_dofs);
        extra_terms.test_compute_magnitude_sq_change_energy(mesh, abars, test_edge_dofs);

        polyscope::init();

        surface_mesh = polyscope::registerSurfaceMesh("Current mesh", fold_V, F);
        std::vector<Eigen::Vector3d> face_edge_midpts = {};
        for (int i = 0; i < mesh.nFaces(); i++) {
            for (int j = 0; j < 3; j++) {
                int eid = mesh.faceEdge(i, j);
                Eigen::Vector3d midpt =
                    (fold_V.row(mesh.edgeVertex(eid, 0)) + fold_V.row(mesh.edgeVertex(eid, 1))) / 2.0;
                face_edge_midpts.push_back(midpt);
            }
        }
        pt_mesh = polyscope::registerPointCloud("Face edge midpoints", face_edge_midpts);

        igl::writeOBJ("rest_mesh.obj", origV, F);
        igl::writeOBJ("fold_mesh.obj", fold_V, F);

        polyscope::state::userCallback = [&]() {
            if (ImGui::Button("Measure Fold Energy", ImVec2(-1, 0))) {
                curenergies = measureFoldEnergy(mesh, origV, fold_V, thickness, lame_alpha, lame_beta);
                log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| "
                    << std::setw(25) << curenergies.quadratic_bending << "| " << std::setw(20) << curenergies.stvk
                    << "| " << std::setw(20) << curenergies.stvk_s1_dir_sin << "| " << std::setw(20)
                    << curenergies.stvk_s1_dir_tan << "| " << std::setw(20) << curenergies.stvk_s2_dir_sin << "| "
                    << std::setw(20) << curenergies.stvk_s2_dir_tan << "| " << std::setw(20)
                    << curenergies.stvk_general_dir << std::endl;
            }
        };

        polyscope::show();

        // curenergies = measureFoldEnergy(mesh, origV, fold_V, thickness, lame_alpha, lame_beta);
        // log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| " << std::setw(25)
        //         << curenergies.quadratic_bending << "| " << std::setw(20) << curenergies.stvk << "| " <<
        //         std::setw(20)
        //         << curenergies.stvk_s1_dir_sin << "| " << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
        //         << std::setw(20) << curenergies.stvk_s2_dir_sin << "| " << std::setw(20) <<
        //         curenergies.stvk_s2_dir_tan
        //         << std::endl;
    }
}
