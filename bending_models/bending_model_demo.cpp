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
#include "../include/ExtraEnergyTermsBase.h"
#include "../include/ExtraEnergyTermsGeneralFormulation.h"
#include "../include/ExtraEnergyTermsGeneralSinFormulation.h"
#include "../include/ExtraEnergyTermsGeneralTanFormulation.h"
#include "../include/ExtraEnergyTermsSinFormulation.h"
#include "../include/ExtraEnergyTermsTanFormulation.h"

#include "../Optimization/include/NewtonDescent.h"
#include "../src/GeometryDerivatives.h"
#include "igl/boundary_loop.h"
#include "igl/null.h"

#include "make_geometric_shapes/HalfCylinder.h"
#include "make_geometric_shapes/Cylinder.h"
#include "make_geometric_shapes/Sphere.h"
#include "spdlog/fmt/bundled/chrono.h"

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

#include <CLI/CLI.hpp>

polyscope::SurfaceMesh* surface_mesh;
polyscope::PointCloud* pt_mesh;

bool is_include_III = true;

void lameParameters(double youngs, double poisson, double& alpha, double& beta) {
    alpha = youngs * poisson / (1.0 - poisson * poisson);
    beta = youngs / 2.0 / (1.0 + poisson);
}

struct StVKDirEnergy {
    double membrane_energy = 0;
    double bending_energy = 0;
    double mag_energy = 0;
    double perp_energy = 0;
};

struct Energies {
    double exact = 0;                           // the theoretic II
    double quadratic_bending = 0;               // quadratic bending
    double stvk_membrane = 0;                   // stvk membrane
    double stvk_bending = 0;                    // stvk bending with fixed edge normal (average adjacent face normals)
    StVKDirEnergy stvk_s1_dir_sin;              // stvk bending with s1 direction, sin formulation
    StVKDirEnergy stvk_s1_dir_tan;              // stvk bending with s1 direction, tan formulation
    StVKDirEnergy stvk_s2_dir_sin;              // stvk bending with s2 direction, sin formulation
    StVKDirEnergy stvk_s2_dir_tan;              // stvk bending with s2 direction, tan formulation
    StVKDirEnergy stvk_general_dir;             // stvk bending with general direction
    StVKDirEnergy stvk_s1_compressive_dir;      // stvk bending with s1 compressive direction
};

enum class SFFGeneralType {
    kGeneralFormulation = 0,
    kS2SinFormulation = 1,
    kS2TanFormulation = 2,
};

enum class SFFUnitS1Type {
    kSinFormulation = 0,
    kTanFormulation = 1,
};

enum class MeshType {
    MT_CYLINDER_IRREGULAR,
    MT_CYLINDER_REGULAR,
    MT_SPHERE,
};

std::vector<Eigen::Vector3d> get_face_edge_normal_vectors(const Eigen::MatrixXd& cur_pos,
                                                          const LibShell::MeshConnectivity& mesh,
                                                          const Eigen::VectorXd& edge_dofs,
                                                          SFFGeneralType& sff_general_type) {
    std::vector<Eigen::Vector3d> face_edge_normals = {};
    int nfaces = mesh.nFaces();

    for (int i = 0; i < nfaces; i++) {
        switch (sff_general_type) {
            case SFFGeneralType::kGeneralFormulation: {
                std::vector<Eigen::Vector3d> general_edge_normals =
                    LibShell::MidedgeAngleGeneralFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
                for (int j = 0; j < 3; j++) {
                    face_edge_normals.push_back(general_edge_normals[j]);
                }
                break;
            }
            case SFFGeneralType::kS2SinFormulation: {
                std::vector<Eigen::Vector3d> general_edge_normals =
                    LibShell::MidedgeAngleGeneralSinFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
                for (int j = 0; j < 3; j++) {
                    face_edge_normals.push_back(general_edge_normals[j]);
                }
                break;
            }
            case SFFGeneralType::kS2TanFormulation: {
                std::vector<Eigen::Vector3d> general_edge_normals =
                    LibShell::MidedgeAngleGeneralTanFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
                for (int j = 0; j < 3; j++) {
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

std::vector<Eigen::Vector3d> get_midsurf_face_edge_normal_vectors(const Eigen::MatrixXd& cur_pos,
                                                                  const LibShell::MeshConnectivity& mesh) {
    std::vector<Eigen::Vector3d> face_edge_normals = {};
    int nfaces = mesh.nFaces();

    for (int i = 0; i < nfaces; i++) {
        Eigen::Vector3d b0 = cur_pos.row(mesh.faceVertex(i, 1)) - cur_pos.row(mesh.faceVertex(i, 0));
        Eigen::Vector3d b1 = cur_pos.row(mesh.faceVertex(i, 2)) - cur_pos.row(mesh.faceVertex(i, 0));
        Eigen::Vector3d nf = b0.cross(b1);
        nf.normalize();
        for (int j = 0; j < 3; j++) {
            face_edge_normals.push_back(nf);
        }
    }
    return face_edge_normals;
}

void optimizeEdgeDOFs(ShellEnergy& energy,
                      const std::vector<Eigen::Matrix2d>& abars,
                      const Eigen::MatrixXd& cur_pos,
                      const LibShell::MeshConnectivity& mesh,
                      const Eigen::VectorXd& edge_area,
                      Eigen::VectorXd& edgeDOFs,
                      std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms = nullptr,
                      std::unordered_set<int>* fixed_edge_dofs = nullptr) {
    double tol = 1e-5;
    int nposdofs = cur_pos.rows() * 3;
    int nedgedofs = edgeDOFs.size();

    std::vector<Eigen::Triplet<double>> Pcoeffs, PE_coeffs;
    int row = 0;
    for (int i = 0; i < nedgedofs; i++) {
        if(!fixed_edge_dofs || fixed_edge_dofs->count(i) == 0) {
            Pcoeffs.push_back({row, nposdofs + i, 1.0});
            PE_coeffs.push_back({row, i, 1.0});
            row++;
        }
    }
    Eigen::SparseMatrix<double> P(row, nposdofs + nedgedofs), PE(row, nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());
    PE.setFromTriplets(PE_coeffs.begin(), PE_coeffs.end());

    Eigen::SparseMatrix<double> PT = P.transpose(), PET = PE.transpose();

    Eigen::VectorXd fixed_edge_dofs_val = Eigen::VectorXd::Zero(nedgedofs);
    if(fixed_edge_dofs) {
        for(auto& id : *fixed_edge_dofs) {
            fixed_edge_dofs_val(id) = edgeDOFs(id);
        }
    }

    auto convert_var_2_edge_dofs = [&](const Eigen::VectorXd& var) {
        return PET * var + fixed_edge_dofs_val;
    };

    auto convert_edge_dofs_2_var = [&](const Eigen::VectorXd& edge_dofs) {
        return PE * edge_dofs;
    };

    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double total_energy = 0;

        auto edge_dofs = convert_var_2_edge_dofs(var);

        double elastic_energy = energy.elasticEnergy(
            cur_pos, edge_dofs, false, true, grad, hessian ? &hessian_triplets : nullptr,
            psd_proj ? LibShell::HessianProjectType::kMaxZero : LibShell::HessianProjectType::kNone);

        total_energy += elastic_energy;

        if (grad) {
            *grad = P * (*grad);
        }

        if (hessian) {
            hessian->resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            *hessian = P * (*hessian) * PT;
        }

        if (extra_energy_terms || edgeDOFs.size() == 4 * edge_area.size()) {
            Eigen::VectorXd mag_comp_deriv, direct_perp_deriv, III_deriv;
            std::vector<Eigen::Triplet<double>> mag_comp_triplets, direct_perp_triplets, III_triplets;

            double mag_comp = extra_energy_terms->compute_magnitude_compression_energy(
                edge_dofs, mesh, grad ? &mag_comp_deriv : nullptr, hessian ? &mag_comp_triplets : nullptr, psd_proj);
            double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
                cur_pos, edge_dofs, mesh, abars, grad ? &direct_perp_deriv : nullptr,
                hessian ? &direct_perp_triplets : nullptr, psd_proj);
            double III = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, edge_dofs, mesh, abars, grad ? &III_deriv : nullptr,
                hessian ? &III_triplets : nullptr, psd_proj);

            total_energy += mag_comp;
            total_energy += direct_perp;

            if(is_include_III) {
                total_energy += III;
            }


            if (grad) {
                *grad += (PE * mag_comp_deriv);
                *grad += P * direct_perp_deriv;

                if(is_include_III) {
                    *grad += P * III_deriv;
                }

            }
            if (hessian) {
                Eigen::SparseMatrix<double> mag_comp_hess, direct_perp_hess, III_hess;

                mag_comp_hess.resize(edge_dofs.size(), edge_dofs.size());
                mag_comp_hess.setFromTriplets(mag_comp_triplets.begin(), mag_comp_triplets.end());
                mag_comp_hess = PE * mag_comp_hess * PET;

                direct_perp_hess.resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
                direct_perp_hess.setFromTriplets(direct_perp_triplets.begin(), direct_perp_triplets.end());
                direct_perp_hess = P * direct_perp_hess * PT;

                III_hess.resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
                III_hess.setFromTriplets(III_triplets.begin(), III_triplets.end());
                III_hess = P * III_hess * PT;

                *hessian += mag_comp_hess;

                *hessian += direct_perp_hess;

                if(is_include_III) {
                    *hessian += III_hess;
                }

            }
        }

        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "------------------------ At beginning -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, edgeDOFs, false, true, NULL, NULL) << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(edgeDOFs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, edgeDOFs, mesh, abars,
                                                                                    nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, edgeDOFs, mesh, abars, nullptr,
                                                                                   nullptr, false);
        extra_energy_terms->test_compute_thirdFundamentalForm_energy(mesh, abars, cur_pos, edgeDOFs);
        extra_energy_terms->test_compute_vector_perp_tangent_energy(mesh, abars, cur_pos, edgeDOFs);

        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;
    }

    Eigen::VectorXd x = convert_edge_dofs_2_var(edgeDOFs);

    OptSolver::TestFuncGradHessian(obj_func, x);

    OptSolver::NewtonSolver(obj_func, find_max_step, x, 1000, tol, 1e-15, 1e-15, true, true, true);

    edgeDOFs = convert_var_2_edge_dofs(x);

    std::cout << "------------------------ At end -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, edgeDOFs, false, true, NULL, NULL) << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(edgeDOFs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, edgeDOFs, mesh, abars,
                                                                                    nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, edgeDOFs, mesh, abars, nullptr,
                                                                                   nullptr, false);

        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;
    }
}

void optimizeFullDOFs(ShellEnergy& energy,
                      const std::vector<Eigen::Matrix2d>& abars,
                      const LibShell::MeshConnectivity& mesh,
                      const Eigen::VectorXd& edge_area,
                      Eigen::MatrixXd& cur_pos,
                      Eigen::VectorXd& cur_edge_dofs,
                      std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms = nullptr,
                      std::unordered_set<int>* fixed_vert_dofs = nullptr,
                      std::unordered_set<int>* fixed_edge_dofs = nullptr) {
    double tol = 1e-5;
    int num_iter = 200;
    int nposdofs = cur_pos.rows() * 3;
    int nedgedofs = cur_edge_dofs.size();
    int nfulldofs = nposdofs + nedgedofs;

    std::vector<Eigen::Triplet<double>> Pcoeffs;
    int row = 0;
    for (int i = 0; i < nposdofs; i++) {
        if(!fixed_vert_dofs || fixed_vert_dofs->count(i) == 0) {
            Pcoeffs.push_back({row,  i, 1.0});
            row++;
        }
    }
    for (int i = 0; i < nedgedofs; i++) {
        if(!fixed_edge_dofs || fixed_edge_dofs->count(i) == 0) {
            Pcoeffs.push_back({row, nposdofs + i, 1.0});
            row++;
        }
    }
    Eigen::SparseMatrix<double> P(row, nfulldofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    Eigen::SparseMatrix<double> PT = P.transpose();

    Eigen::VectorXd fixed_full_dofs_val = Eigen::VectorXd::Zero(nfulldofs);
    if(fixed_vert_dofs) {
        for(auto& id : *fixed_vert_dofs) {
            fixed_full_dofs_val(id) = cur_pos(id / 3, id % 3);
        }
    }

    if(fixed_edge_dofs) {
        for(auto& id : *fixed_edge_dofs) {
            fixed_full_dofs_val(nposdofs + id) = cur_edge_dofs(id);
        }
    }

    auto separate_full_dofs = [&](const Eigen::VectorXd& full_dofs) {
        Eigen::MatrixXd converted_pos = cur_pos;
        Eigen::VectorXd converted_edge_dofs = cur_edge_dofs;
        for(int i = 0; i < cur_pos.rows(); i++) {
            for(int j = 0; j < 3; j++) {
                converted_pos(i, j) = full_dofs(i * 3 + j);
            }
        }
        for(int j = 0; j < cur_edge_dofs.size(); j++) {
            converted_edge_dofs(j) = full_dofs(nposdofs + j);
        }
        return std::make_pair(converted_pos, converted_edge_dofs);
    };

    auto combine_2_full_dofs = [&](const Eigen::MatrixXd& pos_to_combine, const Eigen::VectorXd& edge_dofs_to_combine) {
        Eigen::VectorXd full_dofs = Eigen::VectorXd::Zero(nfulldofs);
        for(int i = 0; i < cur_pos.rows(); i++) {
            for(int j = 0; j < 3; j++) {
                full_dofs(i * 3 + j) = pos_to_combine(i, j);
            }
        }
        for(int j = 0; j < cur_edge_dofs.size(); j++) {
            full_dofs(nposdofs + j) = edge_dofs_to_combine(j);
        }
        return full_dofs;
    };

    auto convert_var_2_pos_edge_dofs = [&](const Eigen::VectorXd& var) {
        auto full_dofs = PT * var + fixed_full_dofs_val;
        return separate_full_dofs(full_dofs);
    };

    auto convert_pos_edge_dofs_2_var = [&](const Eigen::MatrixXd& pos_to_convert, const Eigen::VectorXd& edge_dofs_to_convert) {
        Eigen::VectorXd full_dofs = combine_2_full_dofs(pos_to_convert, edge_dofs_to_convert);
        Eigen::VectorXd reduced_dofs = P * full_dofs;
        return reduced_dofs;
    };


    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        double total_energy = 0;

        auto [pos, edge_dofs] = convert_var_2_pos_edge_dofs(var);

        double elastic_energy = energy.elasticEnergy(
            pos, edge_dofs, true, true, grad, hessian ? &hessian_triplets : nullptr,
            psd_proj ? LibShell::HessianProjectType::kMaxZero : LibShell::HessianProjectType::kNone);

        total_energy += elastic_energy;

        if (grad) {
            *grad = P * (*grad);
        }

        if (hessian) {
            hessian->resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            *hessian = P * (*hessian) * PT;
        }

        if (extra_energy_terms || nedgedofs == 4 * edge_area.size()) {
            Eigen::VectorXd mag_comp_deriv, direct_perp_deriv, III_deriv;
            std::vector<Eigen::Triplet<double>> mag_comp_triplets, direct_perp_triplets, III_triplets;

            double mag_comp = extra_energy_terms->compute_magnitude_compression_energy(
            edge_dofs, mesh, grad ? &mag_comp_deriv : nullptr, hessian ? &mag_comp_triplets : nullptr, psd_proj);
            double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
                pos, edge_dofs, mesh, abars, grad ? &direct_perp_deriv : nullptr,
                hessian ? &direct_perp_triplets : nullptr, psd_proj);
            double III = extra_energy_terms->compute_thirdFundamentalForm_energy(pos, edge_dofs, mesh, abars, grad ? &III_deriv : nullptr,
                hessian ? &III_triplets : nullptr, psd_proj);

            total_energy += mag_comp;
            total_energy += direct_perp;

            if(is_include_III) {
                total_energy += III;
            }


            if (grad) {
                Eigen::VectorXd full_mag_comp_deriv = Eigen::VectorXd::Zero(nfulldofs);
                full_mag_comp_deriv.segment(nposdofs, nedgedofs) = mag_comp_deriv;
                *grad += P * (direct_perp_deriv + full_mag_comp_deriv);

                if(is_include_III) {
                    *grad += P * III_deriv;
                }
            }
            if (hessian) {
                Eigen::SparseMatrix<double> mag_comp_hess, direct_perp_hess, III_hess;

                std::vector<Eigen::Triplet<double>> full_mag_comp_triplets;
                for(int i = 0; i < mag_comp_triplets.size(); i++) {
                    full_mag_comp_triplets.push_back(Eigen::Triplet<double>(mag_comp_triplets[i].row() + nposdofs, mag_comp_triplets[i].col() + nposdofs, mag_comp_triplets[i].value()));
                }

                mag_comp_hess.resize(nfulldofs, nfulldofs);
                mag_comp_hess.setFromTriplets(full_mag_comp_triplets.begin(), full_mag_comp_triplets.end());
                mag_comp_hess = P * mag_comp_hess * PT;

                direct_perp_hess.resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
                direct_perp_hess.setFromTriplets(direct_perp_triplets.begin(), direct_perp_triplets.end());
                direct_perp_hess = P * direct_perp_hess * PT;

                III_hess.resize(edge_dofs.size() + 3 * cur_pos.rows(), edge_dofs.size() + 3 * cur_pos.rows());
                III_hess.setFromTriplets(III_triplets.begin(), III_triplets.end());
                III_hess = P * III_hess * PT;

                *hessian += mag_comp_hess;

                *hessian += direct_perp_hess;

                if(is_include_III) {
                    *hessian += III_hess;
                }

            }
        }

        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "------------------------ At beginning -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, true, NULL, NULL) << std::endl
              << "membrane energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL) << std::endl
              << "bending energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL) << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(cur_edge_dofs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, cur_edge_dofs, mesh, abars,
                                                                                    nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, cur_edge_dofs, mesh, abars, nullptr,
                                                                                   nullptr, false);

        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;
    }

    Eigen::VectorXd x = convert_pos_edge_dofs_2_var(cur_pos, cur_edge_dofs);

    OptSolver::TestFuncGradHessian(obj_func, x);

    OptSolver::NewtonSolver(obj_func, find_max_step, x, num_iter, tol, 1e-15, 1e-15, true, true, true);

    std::tie(cur_pos, cur_edge_dofs) = convert_var_2_pos_edge_dofs(x);

    std::cout << "------------------------ At end -------------------------" << std::endl
    << "elastic energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, true, NULL, NULL) << std::endl
    << "membrane energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL) << std::endl
    << "bending energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL) << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(cur_edge_dofs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, cur_edge_dofs, mesh, abars,
                                                                                    nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, cur_edge_dofs, mesh, abars, nullptr,
                                                                                   nullptr, false);
        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;
    }
}

// optimize the S1 models (edge dofs is s1 director angle in S1)
void optimizeUnitS1Model(const LibShell::MeshConnectivity& mesh,
                       const Eigen::MatrixXd& rest_pos,
                       Eigen::MatrixXd& cur_pos,
                       double thickness,
                       double young,
                       double poisson,
                       Energies& result,
                       SFFUnitS1Type sff_unit_s1_type,
                       bool with_gui = true) {
    Eigen::VectorXd zero_s1_dir_edge_dofs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zero_s1_dir_edge_dofs, mesh, rest_pos);
    LibShell::MonolayerRestState s1_dir_rest_state;

    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);
    double shear = young / (2.0 * (1.0 + poisson));

    s1_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s1_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s1_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                        s1_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::secondFundamentalForms(
        mesh, rest_pos, zero_s1_dir_edge_dofs, s1_dir_rest_state.bbars);

    // assume rest flat
    for (int i = 0; i < mesh.nFaces(); i++) {
        s1_dir_rest_state.bbars[i].setZero();
    }
    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(s1_dir_rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }
    Eigen::MatrixXi s1_F(mesh.nFaces(), 3);
    for(int i = 0; i < mesh.nFaces(); i++) {
        for(int j = 0; j < 3; j++) {
            s1_F(i, j) = mesh.faceVertex(i, j);
        }
    }

    std::vector<int> bnd_loop;
    std::unordered_set<int> fixed_vert_set;
    igl::boundary_loop(s1_F, bnd_loop);
    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < rest_pos.rows(); i++) {
        x_min = std::min(x_min, rest_pos(i, 0));
        x_max = std::max(x_max, rest_pos(i, 0));
    }

    for(int i = 0; i < bnd_loop.size(); i++) {
        int vid = bnd_loop[i];
        if(std::abs(rest_pos(vid, 0) - x_min) < 1e-5 ||
           std::abs(rest_pos(vid, 0) - x_max) < 1e-5) {
            fixed_vert_set.insert(vid * 3);
            fixed_vert_set.insert(vid * 3 + 1);
            fixed_vert_set.insert(vid * 3 + 2);
        }
    }


    Eigen::VectorXd s1_dir_edge_dofs = zero_s1_dir_edge_dofs;
    Eigen::MatrixXd s1_dir_cur_pos = cur_pos;
    std::string mesh_name = "S1 Sin";

    std::shared_ptr<ShellEnergy> s1_dir_energy_model;
    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms_s1;

    // used for normal visualization
    SFFGeneralType s2_general_type = SFFGeneralType::kS2SinFormulation;

    switch (sff_unit_s1_type) {
        case SFFUnitS1Type::kSinFormulation: {
            std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
            extra_energy_terms_s1 =
                std::make_shared<LibShell::ExtraEnergyTermsSinFormulation>();
            extra_energy_terms_s1->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
            s1_dir_energy_model = std::make_shared<StVKS1DirectorSinShellEnergy>(mesh, s1_dir_rest_state);

            optimizeFullDOFs(*s1_dir_energy_model, s1_dir_rest_state.abars, mesh, edge_area, s1_dir_cur_pos,
        s1_dir_edge_dofs, extra_energy_terms_s1, &fixed_vert_set);
            double perp_sin = extra_energy_terms_s1->compute_vector_perp_tangent_energy(
                s1_dir_cur_pos, s1_dir_edge_dofs, mesh, s1_dir_rest_state.abars, nullptr, nullptr, false);
            result.stvk_s1_dir_sin.membrane_energy =
                s1_dir_energy_model->elasticEnergy(s1_dir_cur_pos, s1_dir_edge_dofs, true, false, nullptr, nullptr);
            result.stvk_s1_dir_sin.bending_energy =
                s1_dir_energy_model->elasticEnergy(s1_dir_cur_pos, s1_dir_edge_dofs, false, true, nullptr, nullptr);
            result.stvk_s1_dir_sin.perp_energy = perp_sin;
            result.stvk_s1_dir_sin.mag_energy = 0;
            mesh_name = "S1 Sin";
            s2_general_type = SFFGeneralType::kS2SinFormulation;
            break;
        }
        case SFFUnitS1Type::kTanFormulation: {
            std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
            extra_energy_terms_s1 =
            std::make_shared<LibShell::ExtraEnergyTermsTanFormulation>();
            extra_energy_terms_s1->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
            s1_dir_energy_model = std::make_shared<StVKS1DirectorTanShellEnergy>(mesh, s1_dir_rest_state);

            optimizeFullDOFs(*s1_dir_energy_model, s1_dir_rest_state.abars, mesh, edge_area, s1_dir_cur_pos, s1_dir_edge_dofs, extra_energy_terms_s1, &fixed_vert_set);
            double perp_tan = extra_energy_terms_s1->compute_vector_perp_tangent_energy(
                s1_dir_cur_pos, s1_dir_edge_dofs, mesh, s1_dir_rest_state.abars, nullptr, nullptr, false);
            result.stvk_s1_dir_tan.membrane_energy =
                s1_dir_energy_model->elasticEnergy(s1_dir_cur_pos, s1_dir_edge_dofs, true, false, nullptr, nullptr);
            result.stvk_s1_dir_tan.bending_energy =
                s1_dir_energy_model->elasticEnergy(s1_dir_cur_pos, s1_dir_edge_dofs, false, true, nullptr, nullptr);
            result.stvk_s1_dir_tan.perp_energy = perp_tan;
            result.stvk_s1_dir_tan.mag_energy = 0;
            mesh_name = "S1 Tan";
            s2_general_type = SFFGeneralType::kS2TanFormulation;
            break;
        }
        default: {
            break;
        }
    }

    // add the gui if needed
    if (with_gui) {
        // surface mesh
        auto s1_mesh = polyscope::registerSurfaceMesh(mesh_name, s1_dir_cur_pos, s1_F);
        // pt mesh
        std::vector<Eigen::Vector3d> face_edge_midpts = {};
        for (int i = 0; i < mesh.nFaces(); i++) {
            for (int j = 0; j < 3; j++) {
                int eid = mesh.faceEdge(i, j);
                Eigen::Vector3d midpt =
                    (s1_dir_cur_pos.row(mesh.edgeVertex(eid, 0)) + s1_dir_cur_pos.row(mesh.edgeVertex(eid, 1))) / 2.0;
                face_edge_midpts.push_back(midpt);
            }
        }
        auto s1_pt_mesh = polyscope::registerPointCloud(mesh_name + " Face edge midpoints", face_edge_midpts);

        Eigen::VectorXd s1_edge_dofs_extended(2 * mesh.nEdges());
        for (int i = 0; i < mesh.nEdges(); i++) {
            s1_edge_dofs_extended(2 * i) = s1_dir_edge_dofs(i);
            s1_edge_dofs_extended(2 * i + 1) = M_PI_2;
        }
        std::vector<Eigen::Vector3d> s1_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s1_edge_dofs_extended, s2_general_type);

        s1_pt_mesh->addVectorQuantity("S1 Edge Normals", s1_face_edge_normals);

        std::vector<double> bending_s1_scalars;
        switch (sff_unit_s1_type) {
            case SFFUnitS1Type::kSinFormulation: {
                std::shared_ptr<StVKS1DirectorSinShellEnergy> subPtr = std::static_pointer_cast<StVKS1DirectorSinShellEnergy>(s1_dir_energy_model);
                for (int i = 0; i < mesh.nFaces(); i++) {
                    double s1_bending = subPtr->mat_.bendingEnergy(mesh, s1_dir_cur_pos, s1_dir_edge_dofs, s1_dir_rest_state, i, nullptr, nullptr);
                    bending_s1_scalars.push_back(s1_bending);
                }
                break;
            }
            case SFFUnitS1Type::kTanFormulation: {
                std::shared_ptr<StVKS1DirectorTanShellEnergy> subPtr = std::static_pointer_cast<StVKS1DirectorTanShellEnergy>(s1_dir_energy_model);
                for (int i = 0; i < mesh.nFaces(); i++) {
                    double s1_bending = subPtr->mat_.bendingEnergy(mesh, s1_dir_cur_pos, s1_dir_edge_dofs, s1_dir_rest_state, i, nullptr, nullptr);
                    bending_s1_scalars.push_back(s1_bending);
                }
                break;
            }
            default: {
                bending_s1_scalars.resize(mesh.nFaces(), 0);
                break;
            }
        }

        auto s1_bending_plot = s1_mesh->addFaceScalarQuantity("s1 bending", bending_s1_scalars);
        s1_bending_plot->setMapRange({*std::min_element(bending_s1_scalars.begin(), bending_s1_scalars.end()),
                                         *std::max_element(bending_s1_scalars.begin(), bending_s1_scalars.end())});
    }
}

// optimize the general models (edge dofs is the director angle in S2)
void optimizeS2Model(const LibShell::MeshConnectivity& mesh,
                       const Eigen::MatrixXd& rest_pos,
                       Eigen::MatrixXd& cur_pos,
                       double thickness,
                       double young,
                       double poisson,
                       Energies& result,
                       SFFGeneralType sff_general_type,
                       bool with_gui = true) {
    Eigen::VectorXd half_pi_zero_s2_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(half_pi_zero_s2_dir_edge_dofs, mesh, rest_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState s2_dir_rest_state;

    // set uniform thicknesses
    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);
    double shear = young / (2.0 * (1.0 + poisson));
    s2_dir_rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    s2_dir_rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    s2_dir_rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                               s2_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::secondFundamentalForms(
        mesh, rest_pos, half_pi_zero_s2_dir_edge_dofs, s2_dir_rest_state.bbars);

    // assume rest flat
    for (int i = 0; i < mesh.nFaces(); i++) {
        s2_dir_rest_state.bbars[i].setZero();
    }

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(s2_dir_rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    Eigen::MatrixXi s2_F(mesh.nFaces(), 3);
    for(int i = 0; i < mesh.nFaces(); i++) {
        for(int j = 0; j < 3; j++) {
            s2_F(i, j) = mesh.faceVertex(i, j);
        }
    }

    std::vector<int> bnd_loop;
    std::unordered_set<int> fixed_vert_set;
    igl::boundary_loop(s2_F, bnd_loop);
    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < rest_pos.rows(); i++) {
        x_min = std::min(x_min, rest_pos(i, 0));
        x_max = std::max(x_max, rest_pos(i, 0));
    }

    for(int i = 0; i < bnd_loop.size(); i++) {
        int vid = bnd_loop[i];
        if(std::abs(rest_pos(vid, 0) - x_min) < 1e-5 ||
           std::abs(rest_pos(vid, 0) - x_max) < 1e-5) {
            fixed_vert_set.insert(vid * 3);
            fixed_vert_set.insert(vid * 3 + 1);
            fixed_vert_set.insert(vid * 3 + 2);
           }
    }

    Eigen::VectorXd s2_dir_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    Eigen::MatrixXd s2_dir_cur_pos = cur_pos;
    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms;
    std::shared_ptr<ShellEnergy> stvk_s2_dir_energy_model;

    std::string mesh_name = "None";

    switch (sff_general_type) {
        case SFFGeneralType::kS2SinFormulation: {
            std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
            extra_energy_terms =
                std::make_shared<LibShell::ExtraEnergyTermsGeneralSinFormulation>();
            extra_energy_terms->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
            stvk_s2_dir_energy_model = std::make_shared<StVKS2DirectorSinShellEnergy>(mesh, s2_dir_rest_state);

            optimizeFullDOFs(*stvk_s2_dir_energy_model, s2_dir_rest_state.abars, mesh, edge_area, s2_dir_cur_pos,
                             s2_dir_edge_dofs, extra_energy_terms, &fixed_vert_set);
            double s2_dir_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
                s2_dir_cur_pos, s2_dir_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);
            result.stvk_s2_dir_sin.membrane_energy =
                stvk_s2_dir_energy_model->elasticEnergy(s2_dir_cur_pos, s2_dir_edge_dofs, true, false, nullptr, nullptr);
            result.stvk_s2_dir_sin.bending_energy =
                stvk_s2_dir_energy_model->elasticEnergy(s2_dir_cur_pos, s2_dir_edge_dofs, false, true, nullptr, nullptr);
            result.stvk_s2_dir_sin.perp_energy = s2_dir_perp;
            result.stvk_s2_dir_sin.mag_energy = 0;
            mesh_name = "S2 Sin";
            break;
        }
        case SFFGeneralType::kS2TanFormulation: {
            std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
            extra_energy_terms =
                std::make_shared<LibShell::ExtraEnergyTermsGeneralTanFormulation>();
            extra_energy_terms->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
            stvk_s2_dir_energy_model = std::make_shared<StVKS2DirectorTanShellEnergy>(mesh, s2_dir_rest_state);

            optimizeFullDOFs(*stvk_s2_dir_energy_model, s2_dir_rest_state.abars, mesh, edge_area, s2_dir_cur_pos,
                             s2_dir_edge_dofs, extra_energy_terms, &fixed_vert_set);
            double s2_dir_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
                s2_dir_cur_pos, s2_dir_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);
            result.stvk_s2_dir_tan.membrane_energy =
                stvk_s2_dir_energy_model->elasticEnergy(s2_dir_cur_pos, s2_dir_edge_dofs, true, false, nullptr, nullptr);
            result.stvk_s2_dir_tan.bending_energy =
                stvk_s2_dir_energy_model->elasticEnergy(s2_dir_cur_pos, s2_dir_edge_dofs, false, true, nullptr, nullptr);
            result.stvk_s2_dir_tan.perp_energy = s2_dir_perp;
            result.stvk_s2_dir_tan.mag_energy = 0;
            mesh_name = "S2 Tan";
            break;
        }
        case SFFGeneralType::kGeneralFormulation: {
            std::cout << "This case is not covered yet" << std::endl;
            break;
        }
    }

    if (with_gui) {
        // surface mesh
        auto s2_mesh = polyscope::registerSurfaceMesh(mesh_name, s2_dir_cur_pos, s2_F);
        // pt mesh
        std::vector<Eigen::Vector3d> face_edge_midpts = {};
        for (int i = 0; i < mesh.nFaces(); i++) {
            for (int j = 0; j < 3; j++) {
                int eid = mesh.faceEdge(i, j);
                Eigen::Vector3d midpt =
                    (s2_dir_cur_pos.row(mesh.edgeVertex(eid, 0)) + s2_dir_cur_pos.row(mesh.edgeVertex(eid, 1))) / 2.0;
                face_edge_midpts.push_back(midpt);
            }
        }
        auto s2_pt_mesh = polyscope::registerPointCloud(mesh_name + " Face edge midpoints", face_edge_midpts);


        std::vector<Eigen::Vector3d> face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_edge_dofs, sff_general_type);

        s2_pt_mesh->addVectorQuantity("Edge Normals", face_edge_normals);

        std::vector<double> perp_scalars;
        for (int i = 0; i < mesh.nFaces(); i++) {
            perp_scalars.push_back(extra_energy_terms->compute_vector_perp_tangent_energy_perface(
                s2_dir_cur_pos, s2_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
        }

        std::vector<double> bending_s2_scalars;
        switch (sff_general_type) {
            case SFFGeneralType::kS2SinFormulation: {
                std::shared_ptr<StVKS2DirectorSinShellEnergy> subPtr = std::static_pointer_cast<StVKS2DirectorSinShellEnergy>(stvk_s2_dir_energy_model);
                for (int i = 0; i < mesh.nFaces(); i++) {
                    double s2_bending = subPtr->mat_.bendingEnergy(mesh, s2_dir_cur_pos, s2_dir_edge_dofs, s2_dir_rest_state, i, nullptr, nullptr);
                    bending_s2_scalars.push_back(s2_bending);
                }
                break;
            }
            case SFFGeneralType::kS2TanFormulation: {
                std::shared_ptr<StVKS2DirectorTanShellEnergy> subPtr = std::static_pointer_cast<StVKS2DirectorTanShellEnergy>(stvk_s2_dir_energy_model);
                for (int i = 0; i < mesh.nFaces(); i++) {
                    double s2_bending = subPtr->mat_.bendingEnergy(mesh, s2_dir_cur_pos, s2_dir_edge_dofs, s2_dir_rest_state, i, nullptr, nullptr);
                    bending_s2_scalars.push_back(s2_bending);
                }
                break;
            }
            default: {
                bending_s2_scalars.resize(mesh.nFaces(), 0);
                break;
            }
        }

        auto perp_plot = s2_mesh->addFaceScalarQuantity("perp", perp_scalars);
        perp_plot->setMapRange({*std::min_element(perp_scalars.begin(), perp_scalars.end()),
                                *std::max_element(perp_scalars.begin(), perp_scalars.end())});


        auto s2_bending_plot = s2_mesh->addFaceScalarQuantity("s2 bending", bending_s2_scalars);
        s2_bending_plot->setMapRange({*std::min_element(bending_s2_scalars.begin(), bending_s2_scalars.end()),
                                         *std::max_element(bending_s2_scalars.begin(), bending_s2_scalars.end())});
    }
}

Energies meassureEnergyFullModel(const LibShell::MeshConnectivity& mesh,
                       const Eigen::MatrixXd& rest_pos,
                       Eigen::MatrixXd& cur_pos,
                       double thickness,
                       double young,
                       double poisson,
                       bool with_gui = true) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edge_dofs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edge_dofs, mesh, rest_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state;

    // set uniform thicknesses
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);
    double shear = young / (2.0 * (1.0 + poisson));
    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                       rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, rest_pos, edge_dofs,
                                                                                        rest_state.bbars);

    // assume rest flat
    for (int i = 0; i < mesh.nFaces(); i++) {
        rest_state.bbars[i].setZero();
    }

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh, cur_pos, edge_dofs,
                                                                                        cur_bs);

    Eigen::VectorXd rest_edge_dofs = edge_dofs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, rest_pos, rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, true, true, nullptr, nullptr);
    result.stvk_membrane = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_bending = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, false, true, nullptr, nullptr);

    optimizeUnitS1Model(mesh, rest_pos, cur_pos, thickness, young, poisson, result, SFFUnitS1Type::kSinFormulation, with_gui);
    // optimizeUnitS1Model(mesh, rest_pos, cur_pos, thickness, young, poisson, result, SFFUnitS1Type::kTanFormulation, with_gui);
    optimizeS2Model(mesh, rest_pos, cur_pos, thickness, young, poisson, result, SFFGeneralType::kS2SinFormulation, with_gui);
    // optimizeS2Model(mesh, rest_pos, cur_pos, thickness, young, poisson, result, SFFGeneralType::kS2TanFormulation, with_gui);

    return result;
}


Energies meassureEnergy(const LibShell::MeshConnectivity& mesh,
                       const Eigen::MatrixXd& rest_pos,
                       const Eigen::MatrixXd& cur_pos,
                       double thickness,
                       double young,
                       double poisson,
                       bool with_gui = true) {
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
    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);
    double shear = young / (2.0 * (1.0 + poisson));
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

    // assume rest flat
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
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, rest_pos, rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_membrane = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_bending = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, false, true, nullptr, nullptr);

    StVKS1DirectorSinShellEnergy stvk_s1_dir_sin_energy_model(mesh, s1_dir_rest_state);
    StVKS1DirectorTanShellEnergy stvk_s1_dir_tan_energy_model(mesh, s1_dir_rest_state);
    StVKS2DirectorSinShellEnergy stvk_s2_dir_sin_energy_model(mesh, s2_dir_rest_state);
    StVKS2DirectorTanShellEnergy stvk_s2_dir_tan_energy_model(mesh, s2_dir_rest_state);
    StVKGeneralDirectorShellEnergy stvk_general_dir_energy_model(mesh, general_dir_rest_state);

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
    std::shared_ptr<LibShell::ExtraEnergyTermsSinFormulation> extra_energy_terms_s1_sin =
        std::make_shared<LibShell::ExtraEnergyTermsSinFormulation>();
    extra_energy_terms_s1_sin->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
    optimizeEdgeDOFs(stvk_s1_dir_sin_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area,
s1_dir_sin_edge_dofs, extra_energy_terms_s1_sin);
    double perp_sin = extra_energy_terms_s1_sin->compute_vector_perp_tangent_energy(
        cur_pos, s1_dir_sin_edge_dofs, mesh, s1_dir_rest_state.abars, nullptr, nullptr, false);
    result.stvk_s1_dir_sin.membrane_energy =
        stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_s1_dir_sin.bending_energy =
        stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_s1_dir_sin.perp_energy = perp_sin;
    result.stvk_s1_dir_sin.mag_energy = 0;

    Eigen::VectorXd s1_dir_tan_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
    std::shared_ptr<LibShell::ExtraEnergyTermsTanFormulation> extra_energy_terms_s1_tan =
    std::make_shared<LibShell::ExtraEnergyTermsTanFormulation>();
    extra_energy_terms_s1_tan->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
    // optimizeEdgeDOFs(stvk_s1_dir_tan_energy_model, s1_dir_rest_state.abars, cur_pos, mesh, edge_area, s1_dir_tan_edge_dofs, extra_energy_terms_s1_tan);
    double perp_tan = extra_energy_terms_s1_tan->compute_vector_perp_tangent_energy(
        cur_pos, s1_dir_tan_edge_dofs, mesh, s1_dir_rest_state.abars, nullptr, nullptr, false);
    result.stvk_s1_dir_tan.membrane_energy =
        stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_s1_dir_tan.bending_energy =
        stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_s1_dir_tan.perp_energy = perp_tan;
    result.stvk_s1_dir_tan.mag_energy = 0;

    Eigen::VectorXd general_s1_dir_edge_dofs = general_unit_half_pi_zero_dir_edge_dofs;
    SFFGeneralType general_s1_dir_type = SFFGeneralType::kGeneralFormulation;
    std::cout << "============= Optimizing edge direction (General S1 dir, Compressive) =========== " << std::endl;
    std::shared_ptr<LibShell::ExtraEnergyTermsGeneralFormulation> extra_energy_terms_general_s1 =
        std::make_shared<LibShell::ExtraEnergyTermsGeneralFormulation>();
    extra_energy_terms_general_s1->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
    std::unordered_set<int> fixed_edge_dofs;
    for(int i = 0; i < mesh.nEdges(); i++) {
        fixed_edge_dofs.insert(LibShell::MidedgeAngleGeneralFormulation::numExtraDOFs * i + 1);
    }
    // optimizeEdgeDOFs(stvk_general_dir_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area,
                     // general_s1_dir_edge_dofs, extra_energy_terms_general_s1, &fixed_edge_dofs);
    double mag_comp_s1 =
        extra_energy_terms_general_s1->compute_magnitude_compression_energy(general_s1_dir_edge_dofs, mesh, nullptr, nullptr, false);
    double dir_perp_s1 = extra_energy_terms_general_s1->compute_vector_perp_tangent_energy(
        cur_pos, general_s1_dir_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);

    result.stvk_s1_compressive_dir.membrane_energy =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_s1_dir_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_s1_compressive_dir.bending_energy =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_s1_dir_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_s1_compressive_dir.mag_energy = mag_comp_s1;
    result.stvk_s1_compressive_dir.perp_energy = dir_perp_s1;

    Eigen::VectorXd general_dir_edge_dofs = general_unit_half_pi_zero_dir_edge_dofs;
    SFFGeneralType general_type = SFFGeneralType::kGeneralFormulation;
    std::cout << "============= Optimizing edge direction (General dir) =========== " << std::endl;
    std::shared_ptr<LibShell::ExtraEnergyTermsGeneralFormulation> extra_energy_terms =
        std::make_shared<LibShell::ExtraEnergyTermsGeneralFormulation>();
    extra_energy_terms->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
    // optimizeEdgeDOFs(stvk_general_dir_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area,
                     // general_dir_edge_dofs, extra_energy_terms);
    double mag_comp =
        extra_energy_terms->compute_magnitude_compression_energy(general_dir_edge_dofs, mesh, nullptr, nullptr, false);
    double dir_perp = extra_energy_terms->compute_vector_perp_tangent_energy(
        cur_pos, general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);

    result.stvk_general_dir.membrane_energy =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_dir_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_general_dir.bending_energy =
        stvk_general_dir_energy_model.elasticEnergy(cur_pos, general_dir_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_general_dir.mag_energy = mag_comp;
    result.stvk_general_dir.perp_energy = dir_perp;

    Eigen::VectorXd s2_dir_sin_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_sin_general_type = SFFGeneralType::kS2SinFormulation;
    std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
    std::shared_ptr<LibShell::ExtraEnergyTermsGeneralSinFormulation> extra_energy_terms_sin =
        std::make_shared<LibShell::ExtraEnergyTermsGeneralSinFormulation>();
    extra_energy_terms_sin->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);

    optimizeEdgeDOFs(stvk_s2_dir_sin_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area,
                     s2_dir_sin_edge_dofs, extra_energy_terms_sin);
    double s2_sin_dir_perp = extra_energy_terms_sin->compute_vector_perp_tangent_energy(
        cur_pos, s2_dir_sin_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);
    result.stvk_s2_dir_sin.membrane_energy =
        stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_s2_dir_sin.bending_energy =
        stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_s2_dir_sin.perp_energy = s2_sin_dir_perp;
    result.stvk_s2_dir_sin.mag_energy = 0;

    Eigen::VectorXd s2_dir_tan_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    SFFGeneralType s2_tan_general_type = SFFGeneralType::kS2TanFormulation;
    std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
    std::shared_ptr<LibShell::ExtraEnergyTermsGeneralTanFormulation> extra_energy_terms_tan =
        std::make_shared<LibShell::ExtraEnergyTermsGeneralTanFormulation>();
    extra_energy_terms_tan->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);

    // optimizeEdgeDOFs(stvk_s2_dir_tan_energy_model, s2_dir_rest_state.abars, cur_pos, mesh, edge_area,
    //                  s2_dir_tan_edge_dofs, extra_energy_terms_tan);
    double s2_tan_dir_perp = extra_energy_terms_tan->compute_vector_perp_tangent_energy(
        cur_pos, s2_dir_tan_edge_dofs, mesh, s2_dir_rest_state.abars, nullptr, nullptr, false);
    result.stvk_s2_dir_tan.membrane_energy =
        stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, true, false, nullptr, nullptr);
    result.stvk_s2_dir_tan.bending_energy =
        stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, false, true, nullptr, nullptr);
    result.stvk_s2_dir_tan.perp_energy = s2_tan_dir_perp;
    result.stvk_s2_dir_tan.mag_energy = 0;

    if (with_gui) {
        std::vector<Eigen::Vector3d> s2_tan_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_tan_edge_dofs, s2_tan_general_type);
        std::vector<Eigen::Vector3d> s2_sin_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s2_dir_sin_edge_dofs, s2_sin_general_type);
        std::vector<Eigen::Vector3d> general_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, general_dir_edge_dofs, general_type);
        std::vector<Eigen::Vector3d> mid_surface_face_edge_normals =
            get_midsurf_face_edge_normal_vectors(cur_pos, mesh);
        Eigen::VectorXd s1_tan_edge_dofs_extended = s2_dir_tan_edge_dofs, s1_sin_dofs_extended = s2_dir_sin_edge_dofs, s1_ave_dofs_extended = s2_dir_sin_edge_dofs;
        for (int i = 0; i < mesh.nEdges(); i++) {
            s1_tan_edge_dofs_extended(2 * i) = s1_dir_tan_edge_dofs(i);
            s1_tan_edge_dofs_extended(2 * i + 1) = M_PI_2;

            s1_sin_dofs_extended(2 * i) = s1_dir_sin_edge_dofs(i);
            s1_sin_dofs_extended(2 * i + 1) = M_PI_2;

            s1_ave_dofs_extended(2 * i) = zero_s1_dir_edge_dofs(i);
            s1_ave_dofs_extended(2 * i + 1) = M_PI_2;
        }
        std::vector<Eigen::Vector3d> s1_sin_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s1_sin_dofs_extended, s2_sin_general_type);
        std::vector<Eigen::Vector3d> s1_tan_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s1_tan_edge_dofs_extended, s2_tan_general_type);
        std::vector<Eigen::Vector3d> s1_general_face_edge_normals = get_face_edge_normal_vectors(
            cur_pos, mesh, general_s1_dir_edge_dofs, general_s1_dir_type);
        std::vector<Eigen::Vector3d> s1_ave_face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, s1_ave_dofs_extended, s2_sin_general_type);

        pt_mesh->addVectorQuantity("S2 Tan Edge Normals", s2_tan_face_edge_normals);
        pt_mesh->addVectorQuantity("S2 Sin Edge Normals", s2_sin_face_edge_normals);
        pt_mesh->addVectorQuantity("S1 Tan Edge Normals", s1_tan_face_edge_normals);
        pt_mesh->addVectorQuantity("S1 Sin Edge Normals", s1_sin_face_edge_normals);
        pt_mesh->addVectorQuantity("General Edge Normals", general_face_edge_normals);
        pt_mesh->addVectorQuantity("Average Edge Normals", s1_ave_face_edge_normals);
        pt_mesh->addVectorQuantity("Face Normals on edges", mid_surface_face_edge_normals);
        pt_mesh->addVectorQuantity("S1 Comprssive Edge Normals", s1_general_face_edge_normals);

        std::vector<double> mag_comp_scalars, perp_scalars, mag_sq_change_scalars, tan_per_scalars, sin_per_scalars;
        std::vector<double> bending_general_scalars, bending_s1_general_scalars, bending_s1_sin_scalars, bending_s1_tan_scalars, bending_s2_sin_scalars,
            bending_s2_tan_scalars;
        for (int i = 0; i < mesh.nFaces(); i++) {
            mag_comp_scalars.push_back(extra_energy_terms->compute_magnitude_compression_energy_perface(
                general_dir_edge_dofs, mesh, i, nullptr, nullptr, false));
            perp_scalars.push_back(extra_energy_terms->compute_vector_perp_tangent_energy_perface(
                cur_pos, general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
            mag_sq_change_scalars.push_back(extra_energy_terms->compute_magnitude_sq_change_energy_perface(
                general_dir_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
            double general_bending = stvk_general_dir_energy_model.mat_.bendingEnergy(mesh, cur_pos, general_dir_edge_dofs, s2_dir_rest_state, i, nullptr, nullptr);
            bending_general_scalars.push_back(general_bending);

            sin_per_scalars.push_back(extra_energy_terms_sin->compute_vector_perp_tangent_energy_perface(
                cur_pos, s2_dir_sin_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
            double sin_bending = stvk_s2_dir_sin_energy_model.mat_.bendingEnergy(mesh, cur_pos, s2_dir_sin_edge_dofs, s2_dir_rest_state, i, nullptr, nullptr);
            bending_s2_sin_scalars.push_back(sin_bending);

            tan_per_scalars.push_back(extra_energy_terms_tan->compute_vector_perp_tangent_energy_perface(
                cur_pos, s2_dir_tan_edge_dofs, mesh, s2_dir_rest_state.abars, i, nullptr, nullptr, false));
            double tan_bending = stvk_s2_dir_tan_energy_model.mat_.bendingEnergy(mesh, cur_pos, s2_dir_tan_edge_dofs, s2_dir_rest_state, i, nullptr, nullptr);
            bending_s2_tan_scalars.push_back(tan_bending);

            double s1_sin_bending = stvk_s1_dir_sin_energy_model.mat_.bendingEnergy(mesh, cur_pos, s1_dir_sin_edge_dofs, s1_dir_rest_state, i, nullptr, nullptr);
            bending_s1_sin_scalars.push_back(s1_sin_bending);
            double s1_tan_bending = stvk_s1_dir_tan_energy_model.mat_.bendingEnergy(mesh, cur_pos, s1_dir_tan_edge_dofs, s1_dir_rest_state, i, nullptr, nullptr);
            bending_s1_tan_scalars.push_back(s1_tan_bending);

            double s1_general_bending = stvk_general_dir_energy_model.mat_.bendingEnergy(mesh, cur_pos, general_s1_dir_edge_dofs, s2_dir_rest_state, i, nullptr, nullptr);
            bending_s1_general_scalars.push_back(s1_general_bending);
        }

        auto mag_plot = surface_mesh->addFaceScalarQuantity("mag compression", mag_comp_scalars);
        mag_plot->setMapRange({*std::min_element(mag_comp_scalars.begin(), mag_comp_scalars.end()),
                               *std::max_element(mag_comp_scalars.begin(), mag_comp_scalars.end())});

        auto perp_plot = surface_mesh->addFaceScalarQuantity("perp", perp_scalars);
        perp_plot->setMapRange({*std::min_element(perp_scalars.begin(), perp_scalars.end()),
                                *std::max_element(perp_scalars.begin(), perp_scalars.end())});

        auto sin_perp_plot = surface_mesh->addFaceScalarQuantity("sin perp", sin_per_scalars);
        sin_perp_plot->setMapRange({*std::min_element(sin_per_scalars.begin(), sin_per_scalars.end()),
                                    *std::max_element(sin_per_scalars.begin(), sin_per_scalars.end())});

        auto tan_perp_plot = surface_mesh->addFaceScalarQuantity("tan perp", tan_per_scalars);
        tan_perp_plot->setMapRange({*std::min_element(tan_per_scalars.begin(), tan_per_scalars.end()),
                                    *std::max_element(tan_per_scalars.begin(), tan_per_scalars.end())});

        auto general_bending_plot = surface_mesh->addFaceScalarQuantity("general bending", bending_general_scalars);
        general_bending_plot->setMapRange({*std::min_element(bending_general_scalars.begin(), bending_general_scalars.end()),
                                          *std::max_element(bending_general_scalars.begin(), bending_general_scalars.end())});

        auto s2_sin_bending_plot = surface_mesh->addFaceScalarQuantity("s2 sin bending", bending_s2_sin_scalars);
        s2_sin_bending_plot->setMapRange({*std::min_element(bending_s2_sin_scalars.begin(), bending_s2_sin_scalars.end()),
                                         *std::max_element(bending_s2_sin_scalars.begin(), bending_s2_sin_scalars.end())});

        auto s2_tan_bending_plot = surface_mesh->addFaceScalarQuantity("s2 tan bending", bending_s2_tan_scalars);
        s2_tan_bending_plot->setMapRange({*std::min_element(bending_s2_tan_scalars.begin(), bending_s2_tan_scalars.end()),
                                         *std::max_element(bending_s2_tan_scalars.begin(), bending_s2_tan_scalars.end())});

        auto s1_sin_bending_plot = surface_mesh->addFaceScalarQuantity("s1 sin bending", bending_s1_sin_scalars);
        s1_sin_bending_plot->setMapRange({*std::min_element(bending_s1_sin_scalars.begin(), bending_s1_sin_scalars.end()),
                                         *std::max_element(bending_s1_sin_scalars.begin(), bending_s1_sin_scalars.end())});

        auto s1_tan_bending_plot = surface_mesh->addFaceScalarQuantity("s1 tan bending", bending_s1_tan_scalars);
        s1_tan_bending_plot->setMapRange({*std::min_element(bending_s1_tan_scalars.begin(), bending_s1_tan_scalars.end()),
                                         *std::max_element(bending_s1_tan_scalars.begin(), bending_s1_tan_scalars.end())});

        auto s1_general_bending_plot = surface_mesh->addFaceScalarQuantity("s1 general bending", bending_s1_general_scalars);
        s1_general_bending_plot->setMapRange({*std::min_element(bending_s1_general_scalars.begin(), bending_s1_general_scalars.end()),
                                         *std::max_element(bending_s1_general_scalars.begin(), bending_s1_general_scalars.end())});

    }

    return result;
}

Energies measureCylinderEnergy(const LibShell::MeshConnectivity& mesh,
                               const Eigen::MatrixXd& rest_pos,
                               const Eigen::MatrixXd& cur_pos,
                               double thickness,
                               double young,
                               double poisson,
                               double cur_radius,
                               double cur_height,
                               double cur_angle,
                               bool with_gui = true) {
    Energies result = meassureEnergy(mesh, rest_pos, cur_pos, thickness, young, poisson, with_gui);

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

    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm = lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    double area = cur_angle * cur_radius * cur_height;

    result.exact = svnorm * coeff * area;

    return result;
}

Energies measureSphereEnergy(const LibShell::MeshConnectivity& mesh,
                             const Eigen::MatrixXd& cur_pos,
                             double thickness,
                             double young,
                               double poisson,
                             double radius,
                             bool with_gui = true) {
    Energies result = meassureEnergy(mesh, cur_pos, cur_pos, thickness, young, poisson, with_gui);

    // ground truth energy
    Eigen::Matrix2d abar;
    abar.setIdentity();

    Eigen::Matrix2d b;
    b << 1.0 / radius, 0, 0, 1.0 / radius;

    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm = lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = 4.0 * PI * radius * radius;

    result.exact = svnorm * coeff * area;

    return result;
}


Energies measureCylinderEnergyFullModel(const LibShell::MeshConnectivity& mesh,
                               const Eigen::MatrixXd& rest_pos,
                               Eigen::MatrixXd& cur_pos,
                               double thickness,
                               double young,
                               double poisson,
                               double cur_radius,
                               double cur_height,
                               double cur_angle,
                               bool with_gui = true) {
    Energies result = meassureEnergyFullModel(mesh, rest_pos, cur_pos, thickness, young, poisson, with_gui);

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

    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm = lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    double area = cur_angle * cur_radius * cur_height;

    result.exact = svnorm * coeff * area;

    return result;
}

void save_header(std::ofstream& log) {
    // Assuming log is an output stream
    log << std::left;  // Left-align columns
    log << std::setw(15) << "#V"
        << "| " << std::setw(15) << "exact energy"
        // << "| " << std::setw(25) << "Quadratic bending energy"
        << "| " << std::setw(25) << "StVK membrane energy"
        << "| " << std::setw(25) << "StVK bending energy"
        << "| " << std::setw(40) << "StVK_S1_sin membrane energy"
        << "| " << std::setw(40) << "StVK_S1_sin bending energy"
        << "| " << std::setw(30) << "StVK_S1_sin perp energy"
        << "| " << std::setw(30) << "StVK_S1_sin mag energy"

        // << "| " << std::setw(40) << "StVK_S1_tan membrane energy"
        // << "| " << std::setw(40) << "StVK_S1_tan bending energy"
        // << "| " << std::setw(30) << "StVK_S1_tan perp energy"
        // << "| " << std::setw(30) << "StVK_S1_tan mag energy"
        //
        // << "| " << std::setw(40) << "StVK_S1_comp membrane energy"
        // << "| " << std::setw(40) << "StVK_S1_comp bending energy"
        // << "| " << std::setw(30) << "StVK_S1_comp perp energy"
        // << "| " << std::setw(30) << "StVK_S1_comp mag energy"

        << "| " << std::setw(40) << "StVK_S2_sin membrane energy"
        << "| " << std::setw(40) << "StVK_S2_sin bending energy"
        << "| " << std::setw(30) << "StVK_S2_sin perp energy"
        << "| " << std::setw(30) << "StVK_S2_sin mag energy"

        // << "| " << std::setw(40) << "StVK_S2_tan membrane energy"
        // << "| " << std::setw(40) << "StVK_S2_tan bending energy"
        // << "| " << std::setw(30) << "StVK_S2_tan perp energy"
        // << "| " << std::setw(30) << "StVK_S2_tan mag energy"
        //
        // << "| " << std::setw(40) << "StVK_General membrane energy"
        // << "| " << std::setw(40) << "StVK_General bending energy"
        // << "| " << std::setw(30) << "StVK_General perp energy"
        // << "| " << std::setw(30) << "StVK_General mag energy"
        << std::endl;
}

void save_content(std::ofstream& log, int nverts, Energies curenergies) {
    log << std::setw(15) << nverts
        << "| " << std::setw(15) << curenergies.exact
        // << "| " << std::setw(25) << curenergies.quadratic_bending
        << "| " << std::setw(25) << curenergies.stvk_membrane
        << "| " << std::setw(25) << curenergies.stvk_bending
        << "| " << std::setw(40) << curenergies.stvk_s1_dir_sin.membrane_energy
        << "| " << std::setw(40) << curenergies.stvk_s1_dir_sin.bending_energy
        << "| " << std::setw(30) << curenergies.stvk_s1_dir_sin.perp_energy
        << "| " << std::setw(30) << curenergies.stvk_s1_dir_sin.mag_energy

        // << "| " << std::setw(40) << curenergies.stvk_s1_dir_tan.membrane_energy
        // << "| " << std::setw(40) << curenergies.stvk_s1_dir_tan.bending_energy
        // << "| " << std::setw(30) << curenergies.stvk_s1_dir_tan.perp_energy
        // << "| " << std::setw(30) << curenergies.stvk_s1_dir_tan.mag_energy
        //
        // << "| " << std::setw(40) << curenergies.stvk_s1_compressive_dir.membrane_energy
        // << "| " << std::setw(40) << curenergies.stvk_s1_compressive_dir.bending_energy
        // << "| " << std::setw(30) << curenergies.stvk_s1_compressive_dir.perp_energy
        // << "| " << std::setw(30) << curenergies.stvk_s1_compressive_dir.mag_energy

        << "| " << std::setw(40) << curenergies.stvk_s2_dir_sin.membrane_energy
        << "| " << std::setw(40) << curenergies.stvk_s2_dir_sin.bending_energy
        << "| " << std::setw(30) << curenergies.stvk_s2_dir_sin.perp_energy
        << "| " << std::setw(30) << curenergies.stvk_s2_dir_sin.mag_energy

        // << "| " << std::setw(40) << curenergies.stvk_s2_dir_tan.membrane_energy
        // << "| " << std::setw(40) << curenergies.stvk_s2_dir_tan.bending_energy
        // << "| " << std::setw(30) << curenergies.stvk_s2_dir_tan.perp_energy
        // << "| " << std::setw(30) << curenergies.stvk_s2_dir_tan.mag_energy
        //
        // << "| " << std::setw(40) << curenergies.stvk_general_dir.membrane_energy
        // << "| " << std::setw(40) << curenergies.stvk_general_dir.bending_energy
        // << "| " << std::setw(30) << curenergies.stvk_general_dir.perp_energy
        // << "| " << std::setw(30) << curenergies.stvk_general_dir.mag_energy
        << std::endl;
}

struct InputArgs {
    double thickness = 0.01;
    double triangle_area = 0.002;
    MeshType cur_mesh_type = MeshType::MT_SPHERE;
    bool with_gui = false;
};

int main(int argc, char* argv[]) {
    // {
    //     // Test the function
    //     Eigen::MatrixXd V(3, 3), rest_V;
    //     V << 0, 0, 0,
    //         1, 0, 0,
    //         0, 1, 0;
    //     Eigen::MatrixXi F(1, 3);
    //     F << 0, 1, 2;
    //
    //     LibShell::MeshConnectivity mesh(F);
    //     Eigen::VectorXd edge_dofs;
    //     LibShell::MidedgeAngleGeneralSinFormulation::initializeExtraDOFs(edge_dofs, mesh, V);
    //     edge_dofs.setRandom();
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, 0, 0, 1);
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, 0, 1, 0);
    //
    //     LibShell::MidedgeAngleGeneralSinFormulation::test_third_fund_form(mesh, V, edge_dofs, 0);
    //
    //
    //     double r = 0.05;
    //     double h = 0.1;
    //     double a = 0.02;
    //     makeCylinder(false, r, h, a * 6.0 / 4 * M_PI * r * h, rest_V, V,
    //                  F, 6.0 / 4 * M_PI);
    //     mesh = LibShell::MeshConnectivity(F);
    //     LibShell::MidedgeAngleGeneralSinFormulation::initializeExtraDOFs(edge_dofs, mesh, V);
    //     edge_dofs.setRandom();
    //
    //     auto is_interior_face = [&mesh](int face_id) {
    //         for(int i = 0; i < 3; i++) {
    //             int eid = mesh.faceEdge(face_id, i);
    //             if(mesh.edgeFace(eid, 0) == -1 || mesh.edgeFace(eid, 1) == -1) {
    //                 return false;
    //             }
    //         }
    //         return true;
    //     };
    //
    //     int interior_face_id = std::rand() % mesh.nFaces();
    //     while(!is_interior_face(interior_face_id)) {
    //         interior_face_id = std::rand() % mesh.nFaces();
    //     }
    //     int bnd_face_id = std::rand() % mesh.nFaces();
    //     while(is_interior_face(bnd_face_id)) {
    //         bnd_face_id = std::rand() % mesh.nFaces();
    //     }
    //     LibShell::MidedgeAngleGeneralSinFormulation::test_third_fund_form(mesh, V, edge_dofs, interior_face_id);
    //     LibShell::MidedgeAngleGeneralSinFormulation::test_third_fund_form(mesh, V, edge_dofs, bnd_face_id);
    //
    //     // std::cout << "Test interior face: " << interior_face_id << std::endl;
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, interior_face_id, 0, 1);
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, interior_face_id, 2, 1);
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, interior_face_id, 0, 2);
    //     //
    //     // std::cout << "Test boundary face: " << bnd_face_id << std::endl;
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, bnd_face_id, 0, 1);
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, bnd_face_id, 2, 1);
    //     // LibShell::MidedgeAngleGeneralSinFormulation::test_compute_ninj(mesh, V, edge_dofs, bnd_face_id, 0, 2);
    //     return EXIT_SUCCESS;
    // }
    InputArgs args;
    CLI::App app{"Shell Energy Model"};
    app.add_option("-t,--thickness", args.thickness, "Thickness of the shell");
    app.add_option("-a,--triangle_area", args.triangle_area, "Relative triangle area of the mesh");
    app.add_option("-m,--mesh_type", args.cur_mesh_type, "Mesh type: 0 - Cylinder Regular, 1 - Cylinder Irregular, 2 - Sphere");
    app.add_flag("-g,--with_gui", args.with_gui, "With GUI");
    CLI11_PARSE(app, argc, argv);

    double triangle_area = 0.002;
    if(args.triangle_area > 0) {
        triangle_area = args.triangle_area;
    }
    double thickness = 0.01;
    if(args.thickness > 0) {
        thickness = args.thickness;
    }
    MeshType cur_mesh_type = MeshType::MT_CYLINDER_REGULAR;
    if(args.cur_mesh_type == MeshType::MT_CYLINDER_IRREGULAR || args.cur_mesh_type == MeshType::MT_CYLINDER_REGULAR || args.cur_mesh_type == MeshType::MT_SPHERE) {
        cur_mesh_type = args.cur_mesh_type;
    }

    double cylinder_radius = 1;
    double cylinder_height = 1;
    double sphere_radius = 0.05;

    // set up material parameters
    double young = 1e7;
    double poisson = 0;

    Eigen::MatrixXd origV;
    Eigen::MatrixXd rolledV;
    Eigen::MatrixXi F;

    int steps = 5;
    double multiplier = 2;
    std::string log_file_name = "log";
    switch (args.cur_mesh_type) {
        case MeshType::MT_CYLINDER_IRREGULAR:
            log_file_name = "cylinder_irregular";
            break;
        case MeshType::MT_CYLINDER_REGULAR:
            log_file_name = "cylinder_regular";
            break;
        case MeshType::MT_SPHERE:
            log_file_name = "sphere";
            break;
    }
    std::string log_file_name_full = log_file_name + "_full.txt";
    log_file_name += ".txt";
    std::ofstream log(log_file_name);
    save_header(log);

    std::ofstream log_full(log_file_name_full);
    save_header(log_full);

    if (args.cur_mesh_type == MeshType::MT_SPHERE) {
        makeSphere(sphere_radius, triangle_area * 4 * M_PI * sphere_radius * sphere_radius, origV, F);
        LibShell::MeshConnectivity mesh(F);
        rolledV = origV;

        if(args.with_gui) {
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
                if (ImGui::InputDouble("Triangle Area", &triangle_area)) {
                    if (triangle_area <= 0) {
                        triangle_area = 0.02;
                    }
                }

                if (ImGui::InputDouble("Thickness", &thickness)) {
                    if (thickness <= 0) {
                        thickness = 0.001;
                    }
                }

                ImGui::Checkbox("Include III", &is_include_III);

                if (ImGui::Button("Remake Sphere", ImVec2(-1, 0))) {
                    double actual_triangle_area = triangle_area * 4 * M_PI * sphere_radius * sphere_radius;
                    makeSphere(sphere_radius, actual_triangle_area, origV, F);
                    rolledV = origV;
                    mesh = LibShell::MeshConnectivity(F);
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
                }

                if (ImGui::Button("Measure Energy", ImVec2(-1, 0))) {
                    auto curenergies = measureSphereEnergy(mesh, origV, thickness, young, poisson, sphere_radius);
                    save_content(log, origV.rows(), curenergies);
                }
            };

            polyscope::show();
        } else {
            for (int step = 0; step < steps; step++) {
                double actual_triangle_area = triangle_area * 4 * M_PI * sphere_radius * sphere_radius;
                makeSphere(sphere_radius, actual_triangle_area, origV, F);
                rolledV = origV;
                mesh = LibShell::MeshConnectivity(F);

                std::cout << "Mesh generated: #V: " << rolledV.rows() << ", #F: " << F.rows() << std::endl;

                std::stringstream ss;
                ss << "sphere_ " << step << ".ply";
                igl::writePLY(ss.str(), origV, F);

                auto curenergies = measureSphereEnergy(mesh, origV, thickness, young, poisson, sphere_radius, false);
                save_content(log, origV.rows(), curenergies);
                triangle_area *= multiplier;
            }
        }
    } else if (cur_mesh_type == MeshType::MT_CYLINDER_IRREGULAR || cur_mesh_type == MeshType::MT_CYLINDER_REGULAR) {
        if(args.with_gui) {
            makeCylinder(cur_mesh_type == MeshType::MT_CYLINDER_REGULAR, cylinder_radius, cylinder_height, triangle_area * 6.0 / 4 * M_PI * cylinder_radius * cylinder_height, origV, rolledV,
                     F, 6.0 / 4 * M_PI);
            LibShell::MeshConnectivity mesh(F);

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
                if (ImGui::InputDouble("Triangle Area", &triangle_area)) {
                    if (triangle_area <= 0) {
                        triangle_area = 0.02;
                    }
                }

                if (ImGui::InputDouble("Thickness", &thickness)) {
                    if (thickness <= 0) {
                        thickness = 0.001;
                    }
                }

                ImGui::Checkbox("Include III", &is_include_III);

                if (ImGui::Button("Remake Cylinder", ImVec2(-1, 0))) {
                    double actual_triangle_area = triangle_area * 6.0 / 4 * M_PI * cylinder_radius * cylinder_height;
                    makeCylinder(cur_mesh_type == MeshType::MT_CYLINDER_REGULAR, cylinder_radius, cylinder_height, actual_triangle_area, origV, rolledV,
                             F, 6.0 / 4 * M_PI);
                    mesh = LibShell::MeshConnectivity(F);
                    surface_mesh = polyscope::registerSurfaceMesh("Current mesh", rolledV, F);
                    face_edge_midpts.clear();
                    for (int i = 0; i < mesh.nFaces(); i++) {
                        for (int j = 0; j < 3; j++) {
                            int eid = mesh.faceEdge(i, j);
                            Eigen::Vector3d midpt =
                                (rolledV.row(mesh.edgeVertex(eid, 0)) + rolledV.row(mesh.edgeVertex(eid, 1))) / 2.0;
                            face_edge_midpts.push_back(midpt);
                        }
                    }
                    pt_mesh = polyscope::registerPointCloud("Face edge midpoints", face_edge_midpts);
                }

                if (ImGui::Button("Measure Energy", ImVec2(-1, 0))) {
                    auto curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, young, poisson, cylinder_radius, cylinder_height, 6.0 / 4 * M_PI, true);

                    save_content(log, origV.rows(), curenergies);
                }
                if (ImGui::Button("Measure Full Energy", ImVec2(-1, 0))) {
                    auto newV = rolledV;
                    auto curenergies = measureCylinderEnergyFullModel(mesh, origV, newV, thickness, young, poisson, cylinder_radius, cylinder_height, 6.0 / 4 * M_PI, true);

                    save_content(log_full, origV.rows(), curenergies);
                }
            };

            polyscope::show();
        } else {
            double cur_angle = 6.0 / 4 * M_PI;
            double init_triangle_area = triangle_area;
            for (int step = 0; step < steps; step++) {
                double actual_triangle_area = triangle_area * cur_angle * cylinder_radius * cylinder_height;
                makeCylinder(cur_mesh_type == MeshType::MT_CYLINDER_REGULAR, cylinder_radius, cylinder_height, actual_triangle_area, origV, rolledV,
                         F, cur_angle);
                std::cout << "Mesh generated: #V: " << rolledV.rows() << ", #F: " << F.rows() << std::endl;
                LibShell::MeshConnectivity mesh(F);
                auto curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, young, poisson, cylinder_radius, cylinder_height, cur_angle, false);
                save_content(log, origV.rows(), curenergies);

                triangle_area *= multiplier;
            }

            // optimize for the full model
            triangle_area = init_triangle_area;
            for (int step = 0; step < steps; step++) {
                double actual_triangle_area = triangle_area * cur_angle * cylinder_radius * cylinder_height;
                makeCylinder(cur_mesh_type == MeshType::MT_CYLINDER_REGULAR, cylinder_radius, cylinder_height, actual_triangle_area, origV, rolledV,
                         F, cur_angle);
                std::cout << "Mesh generated: #V: " << rolledV.rows() << ", #F: " << F.rows() << std::endl;
                LibShell::MeshConnectivity mesh(F);

                auto cur_full_energies = measureCylinderEnergyFullModel(mesh, origV, rolledV, thickness, young, poisson, cylinder_radius, cylinder_height, cur_angle, false);
                save_content(log_full, origV.rows(), cur_full_energies);
                triangle_area *= multiplier;
            }
        }
    }
}
