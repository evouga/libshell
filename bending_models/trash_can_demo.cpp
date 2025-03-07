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

double start_xmin = 0;
double start_xmax = 1;

int cur_frame_idx = 0; // Track the current frame being displayed

std::vector<int> left_side_vids;
std::vector<int> right_side_vids;

enum class SFFModelType { kS1Sin, kS2Sin };

bool is_include_III = true;

void lameParameters(double youngs, double poisson, double& alpha, double& beta) {
    alpha = youngs * poisson / (1.0 - poisson * poisson);
    beta = youngs / 2.0 / (1.0 + poisson);
}

std::vector<Eigen::Vector3d> get_face_edge_normal_vectors(const Eigen::MatrixXd& cur_pos,
                                                          const LibShell::MeshConnectivity& mesh,
                                                          const Eigen::VectorXd& edge_dofs) {
    std::vector<Eigen::Vector3d> face_edge_normals = {};
    int nfaces = mesh.nFaces();

    for (int i = 0; i < nfaces; i++) {
        std::vector<Eigen::Vector3d> general_edge_normals =
            LibShell::MidedgeAngleGeneralSinFormulation::get_face_edge_normals(mesh, cur_pos, edge_dofs, i);
        for (int j = 0; j < 3; j++) {
            face_edge_normals.push_back(general_edge_normals[j]);
        }
    }
    return face_edge_normals;
}

double get_constraint_penalty(const Eigen::MatrixXd& pos, double cur_time, double stiffness, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessian) {
    double penalty = 0;
    int nverts = pos.rows();
    if (deriv) {
        deriv->setZero(3 * nverts);
    }
    if (hessian) {
        hessian->clear();
    }

    double target_x_left = start_xmin + (start_xmax - start_xmin) / 2;
    double target_x_right = start_xmax - (start_xmax - start_xmin) / 2;
    double left_x_target_cur = start_xmin + (target_x_left - start_xmin) * cur_time;
    double right_x_target_cur = start_xmax - (start_xmax - target_x_right) * cur_time;


    for(int i = 0; i < nverts; i++) {
        if(pos(i, 0) < left_x_target_cur) {
            penalty += 0.5 * stiffness * (pos(i, 0) - left_x_target_cur) * (pos(i, 0) - left_x_target_cur);

            if(deriv) {
                (*deriv)(3 * i) += stiffness * (pos(i, 0) - left_x_target_cur);
            }

            if(hessian) {
                hessian->push_back({3 * i, 3 * i, stiffness});
            }
        } else if (pos(i, 0) > right_x_target_cur) {
            penalty += 0.5 * stiffness * (pos(i, 0) - right_x_target_cur) * (pos(i, 0) - right_x_target_cur);

            if(deriv) {
                (*deriv)(3 * i) += stiffness * (pos(i, 0) - right_x_target_cur);
            }

            if(hessian) {
                hessian->push_back({3 * i, 3 * i, stiffness});
            }
        }
    }

    return penalty;
}

std::unordered_map<int, double> build_boundary_fixed_vertex_map(const Eigen::MatrixXd& pos,
                                                                const std::vector<int>& left_side_vids,
                                                                const std::vector<int>& right_side_vids,
                                                                double cur_time) {
    // move vertex form start pos to target pos from 0 to 1
    double target_x_min = start_xmin + (start_xmax - start_xmin) / 2;
    double target_x_max = start_xmax - (start_xmax - start_xmin) / 2;

    std::unordered_map<int, double> fixed_vertex_dofs;

    // left side (cos(0) = 1), from ymax to target_y
    for (int i = 0; i < left_side_vids.size(); i++) {
        int vid = left_side_vids[i];
        double x_target = start_xmax - (start_xmax - target_x_max) * cur_time;
        fixed_vertex_dofs[3 * vid + 0] = x_target;
        fixed_vertex_dofs[3 * vid + 1] = pos(vid, 1);
        fixed_vertex_dofs[3 * vid + 2] = pos(vid, 2);
    }

    // right side, from ymin to target_y
    for (int i = 0; i < right_side_vids.size(); i++) {
        int vid = right_side_vids[i];
        double x_target = start_xmin + (target_x_min - start_xmin) * cur_time;
        fixed_vertex_dofs[3 * vid + 0] = x_target;
        fixed_vertex_dofs[3 * vid + 1] = pos(vid, 1);
        fixed_vertex_dofs[3 * vid + 2] = pos(vid, 2);
    }

    return fixed_vertex_dofs;
}

std::pair<std::vector<int>, std::vector<int>> get_left_right_sides(const Eigen::MatrixXd& rest_pos,
                                                                   const LibShell::MeshConnectivity& mesh) {
    std::vector<int> bnd_loop, left_side, right_side;
    Eigen::MatrixXi F(mesh.nFaces(), 3);
    for (int i = 0; i < mesh.nFaces(); i++) {
        for (int j = 0; j < 3; j++) {
            F(i, j) = mesh.faceVertex(i, j);
        }
    }

    igl::boundary_loop(F, bnd_loop);
    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < rest_pos.rows(); i++) {
        x_min = std::min(x_min, rest_pos(i, 0));
        x_max = std::max(x_max, rest_pos(i, 0));
    }

    for (int i = 0; i < bnd_loop.size(); i++) {
        int vid = bnd_loop[i];
        if (std::abs(rest_pos(vid, 0) - x_min) < 1e-5) {
            left_side.push_back(vid);
        } else if (std::abs(rest_pos(vid, 0) - x_max) < 1e-5) {
            right_side.push_back(vid);
        }
    }

    return std::make_pair(left_side, right_side);
}

void optimizeFullDOFs(ShellEnergy& energy,
                      const std::vector<Eigen::Matrix2d>& abars,
                      const LibShell::MeshConnectivity& mesh,
                      const Eigen::VectorXd& edge_area,
                      Eigen::MatrixXd& cur_pos,
                      Eigen::VectorXd& cur_edge_dofs,
                      std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms = nullptr,
                      const std::unordered_map<int, double>* fixed_vert_dofs = nullptr,
                      const std::unordered_set<int>* fixed_edge_dofs = nullptr,
                      std::function<double(const Eigen::MatrixXd&, Eigen::VectorXd*, std::vector<Eigen::Triplet<double>>* )> constraint_penalty = nullptr) {
    double tol = 1e-5;
    int num_iter = 200;
    int nposdofs = cur_pos.rows() * 3;
    int nedgedofs = cur_edge_dofs.size();
    int nfulldofs = nposdofs + nedgedofs;

    std::vector<Eigen::Triplet<double>> Pcoeffs;
    int row = 0;
    for (int i = 0; i < nposdofs; i++) {
        if (!fixed_vert_dofs || fixed_vert_dofs->count(i) == 0) {
            Pcoeffs.push_back({row, i, 1.0});
            row++;
        }
    }
    for (int i = 0; i < nedgedofs; i++) {
        if (!fixed_edge_dofs || fixed_edge_dofs->count(i) == 0) {
            Pcoeffs.push_back({row, nposdofs + i, 1.0});
            row++;
        }
    }
    Eigen::SparseMatrix<double> P(row, nfulldofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    Eigen::SparseMatrix<double> PT = P.transpose();

    Eigen::VectorXd fixed_full_dofs_val = Eigen::VectorXd::Zero(nfulldofs);
    if (fixed_vert_dofs) {
        for (auto& id : *fixed_vert_dofs) {
            fixed_full_dofs_val(id.first) = id.second;
        }
    }

    if (fixed_edge_dofs) {
        for (auto& id : *fixed_edge_dofs) {
            fixed_full_dofs_val(nposdofs + id) = cur_edge_dofs(id);
        }
    }

    auto separate_full_dofs = [&](const Eigen::VectorXd& full_dofs) {
        Eigen::MatrixXd converted_pos = cur_pos;
        Eigen::VectorXd converted_edge_dofs = cur_edge_dofs;
        for (int i = 0; i < cur_pos.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                converted_pos(i, j) = full_dofs(i * 3 + j);
            }
        }
        for (int j = 0; j < cur_edge_dofs.size(); j++) {
            converted_edge_dofs(j) = full_dofs(nposdofs + j);
        }
        return std::make_pair(converted_pos, converted_edge_dofs);
    };

    auto combine_2_full_dofs = [&](const Eigen::MatrixXd& pos_to_combine, const Eigen::VectorXd& edge_dofs_to_combine) {
        Eigen::VectorXd full_dofs = Eigen::VectorXd::Zero(nfulldofs);
        for (int i = 0; i < cur_pos.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                full_dofs(i * 3 + j) = pos_to_combine(i, j);
            }
        }
        for (int j = 0; j < cur_edge_dofs.size(); j++) {
            full_dofs(nposdofs + j) = edge_dofs_to_combine(j);
        }
        return full_dofs;
    };

    auto convert_var_2_pos_edge_dofs = [&](const Eigen::VectorXd& var) {
        auto full_dofs = PT * var + fixed_full_dofs_val;
        return separate_full_dofs(full_dofs);
    };

    auto convert_pos_edge_dofs_2_var = [&](const Eigen::MatrixXd& pos_to_convert,
                                           const Eigen::VectorXd& edge_dofs_to_convert) {
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

        Eigen::VectorXd penalty_grad;
        std::vector<Eigen::Triplet<double>> penalty_hessian;
        double penalty = constraint_penalty(pos, grad ? &penalty_grad : nullptr, hessian ? &penalty_hessian : nullptr);

        total_energy += penalty;

        if (grad) {
            grad->segment(0, 3 * pos.rows()) += penalty_grad;
            *grad = P * (*grad);
        }

        if (hessian) {
            hessian_triplets.insert(hessian_triplets.end(), penalty_hessian.begin(), penalty_hessian.end());
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
            double III = extra_energy_terms->compute_thirdFundamentalForm_energy(
                pos, edge_dofs, mesh, abars, grad ? &III_deriv : nullptr, hessian ? &III_triplets : nullptr, psd_proj);

            total_energy += mag_comp;
            total_energy += direct_perp;

            if (is_include_III) {
                total_energy += III;
            }

            if (grad) {
                Eigen::VectorXd full_mag_comp_deriv = Eigen::VectorXd::Zero(nfulldofs);
                full_mag_comp_deriv.segment(nposdofs, nedgedofs) = mag_comp_deriv;
                *grad += P * (direct_perp_deriv + full_mag_comp_deriv);

                if (is_include_III) {
                    *grad += P * III_deriv;
                }
            }
            if (hessian) {
                Eigen::SparseMatrix<double> mag_comp_hess, direct_perp_hess, III_hess;

                std::vector<Eigen::Triplet<double>> full_mag_comp_triplets;
                for (int i = 0; i < mag_comp_triplets.size(); i++) {
                    full_mag_comp_triplets.push_back(Eigen::Triplet<double>(mag_comp_triplets[i].row() + nposdofs,
                                                                            mag_comp_triplets[i].col() + nposdofs,
                                                                            mag_comp_triplets[i].value()));
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

                if (is_include_III) {
                    *hessian += III_hess;
                }
            }
        }

        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "------------------------ At beginning -------------------------" << std::endl
              << "elastic energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, true, NULL, NULL) << std::endl
              << "membrane energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL)
              << std::endl
              << "bending energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL)
              << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(cur_edge_dofs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, cur_edge_dofs, mesh, abars,
                                                                                    nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, cur_edge_dofs, mesh, abars,
                                                                                  nullptr, nullptr, false);

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
              << "membrane energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, true, false, NULL, NULL)
              << std::endl
              << "bending energy: " << energy.elasticEnergy(cur_pos, cur_edge_dofs, false, true, NULL, NULL)
              << std::endl;

    if (extra_energy_terms) {
        double mag_comp =
            extra_energy_terms->compute_magnitude_compression_energy(cur_edge_dofs, mesh, nullptr, nullptr, false);
        double direct_perp = extra_energy_terms->compute_vector_perp_tangent_energy(cur_pos, cur_edge_dofs, mesh, abars,
                                                                                    nullptr, nullptr, false);
        double III_term = extra_energy_terms->compute_thirdFundamentalForm_energy(cur_pos, cur_edge_dofs, mesh, abars,
                                                                                  nullptr, nullptr, false);
        std::cout << "||m^2 - 1||^2: " << mag_comp << std::endl;
        std::cout << "direct perp: " << direct_perp << std::endl;
        std::cout << "III: " << III_term << std::endl;
    }
}

std::pair<std::shared_ptr<ShellEnergy>, std::shared_ptr<LibShell::ExtraEnergyTermsBase>> initialization(const LibShell::MeshConnectivity& mesh,
                    const Eigen::MatrixXd& rest_pos,
                    double thickness,
                    double young,
                    double poisson,
                    Eigen::VectorXd& edge_dofs,
                    LibShell::MonolayerRestState& rest_state,
                    Eigen::VectorXd& edge_area,
                    SFFModelType model_type) {
    rest_state.abars.clear();
    rest_state.thicknesses.clear();
    rest_state.bbars.clear();
    rest_state.lameAlpha.clear();
    rest_state.lameAlpha.clear();
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);
    double lame_alpha, lame_beta;
    lameParameters(young, poisson, lame_alpha, lame_beta);
    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::firstFundamentalForms(mesh, rest_pos,
                                                                                        rest_state.abars);
    rest_state.bbars = rest_state.abars;
    for(auto& mat : rest_state.bbars) {
        mat.setZero();
    }

    edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if (fid != -1) {
                edge_area[i] += std::sqrt(rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    double shear = young / (2.0 * (1.0 + poisson));

    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms;
    std::shared_ptr<ShellEnergy> stvk_dir_energy_model;

    switch (model_type) {
        case SFFModelType::kS1Sin: {
            LibShell::MidedgeAngleSinFormulation::initializeExtraDOFs(edge_dofs, mesh, rest_pos);
            extra_energy_terms = std::make_shared<LibShell::ExtraEnergyTermsSinFormulation>();
            extra_energy_terms->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
            stvk_dir_energy_model = std::make_shared<StVKS1DirectorSinShellEnergy>(mesh, rest_state);
            break;
        }

        case SFFModelType::kS2Sin: {
            LibShell::MidedgeAngleGeneralSinFormulation::initializeExtraDOFs(edge_dofs, mesh, rest_pos);
            extra_energy_terms = std::make_shared<LibShell::ExtraEnergyTermsGeneralSinFormulation>();
            extra_energy_terms->initialization(rest_pos, mesh, young, shear, thickness, poisson, 3);
            stvk_dir_energy_model = std::make_shared<StVKS2DirectorSinShellEnergy>(mesh, rest_state);
            break;
        }
    }

    return {stvk_dir_energy_model, extra_energy_terms};
}

void processFullModelOneStep(const LibShell::MeshConnectivity& mesh,
                             const LibShell::MonolayerRestState& rest_state,
                             const Eigen::VectorXd& edge_area,
                             const std::shared_ptr<ShellEnergy> stvk_dir_energy_model,
                             const std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms,
                             const double penalty_stiffness,
                             Eigen::MatrixXd& cur_pos,
                             Eigen::VectorXd& cur_edge_dofs,
                             SFFModelType model_type,
                             double current_time = 0) {
    std::unordered_map<int, double> cur_bnd_condition =
        build_boundary_fixed_vertex_map(cur_pos, left_side_vids, right_side_vids, current_time);
    auto constraint_penalty = [&](const Eigen::MatrixXd& pos, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessian) {
        return get_constraint_penalty(pos, current_time, penalty_stiffness, deriv, hessian);
    };
    switch (model_type) {
        case SFFModelType::kS1Sin:
            std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
        break;
        case SFFModelType::kS2Sin:
            std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
        break;
    }
    optimizeFullDOFs(*stvk_dir_energy_model, rest_state.abars, mesh, edge_area, cur_pos, cur_edge_dofs, extra_energy_terms,
                     cur_bnd_condition.empty() ? nullptr : &cur_bnd_condition, nullptr, constraint_penalty);


}

void update_rendering(polyscope::SurfaceMesh* cur_mesh, polyscope::PointCloud* pt_mesh, const std::shared_ptr<ShellEnergy> stvk_dir_energy_model,
                             const std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms,
                      const LibShell::MeshConnectivity& mesh, const LibShell::MonolayerRestState& rest_state, const Eigen::MatrixXd& cur_pos, const Eigen::VectorXd& cur_edge_dofs,
                      SFFModelType model_type) {
    std::vector<Eigen::Vector3d> face_edge_midpts = {};
    for (int i = 0; i < mesh.nFaces(); i++) {
        for (int j = 0; j < 3; j++) {
            int eid = mesh.faceEdge(i, j);
            Eigen::Vector3d midpt =
                (cur_pos.row(mesh.edgeVertex(eid, 0)) + cur_pos.row(mesh.edgeVertex(eid, 1))) / 2.0;
            face_edge_midpts.push_back(midpt);
        }
    }
    pt_mesh->updatePointPositions(face_edge_midpts);
    cur_mesh->updateVertexPositions(cur_pos);

    // draw the edge normals
    Eigen::VectorXd edge_dofs;
    std::string model_name;
    switch (model_type) {
        case SFFModelType::kS1Sin: {
            edge_dofs.resize(2 * mesh.nEdges());
            for (int i = 0; i < mesh.nEdges(); i++) {
                edge_dofs(2 * i) = cur_edge_dofs(i);
                edge_dofs(2 * i + 1) = M_PI_2;
            }
            model_name = "S1 sin";
            break;
        }
        case SFFModelType::kS2Sin: {
            edge_dofs = cur_edge_dofs;
            model_name = "S2 sin";
            break;
        }
        default: {
            return;
        }
    }
    std::vector<Eigen::Vector3d> face_edge_normals =
            get_face_edge_normal_vectors(cur_pos, mesh, edge_dofs);

    // draw energy terms
    std::vector<double> bending_scalars;
    switch (model_type) {
        case SFFModelType::kS1Sin: {
            std::shared_ptr<StVKS1DirectorSinShellEnergy> subPtr = std::static_pointer_cast<StVKS1DirectorSinShellEnergy>(stvk_dir_energy_model);
            for (int i = 0; i < mesh.nFaces(); i++) {
                double s1_bending = subPtr->mat_.bendingEnergy(mesh, cur_pos, edge_dofs, rest_state, i, nullptr, nullptr);
                bending_scalars.push_back(s1_bending);
            }
            break;
        }
        case SFFModelType::kS2Sin: {
            std::shared_ptr<StVKS1DirectorTanShellEnergy> subPtr = std::static_pointer_cast<StVKS1DirectorTanShellEnergy>(stvk_dir_energy_model);
            for (int i = 0; i < mesh.nFaces(); i++) {
                double s1_bending = subPtr->mat_.bendingEnergy(mesh, cur_pos, edge_dofs, rest_state, i, nullptr, nullptr);
                bending_scalars.push_back(s1_bending);
            }
            break;
        }
        default: {
            bending_scalars.resize(mesh.nFaces(), 0);
            break;
        }
    }

    auto bending_plot = surface_mesh->addFaceScalarQuantity("bending", bending_scalars);
    bending_plot->setMapRange({*std::min_element(bending_scalars.begin(), bending_scalars.end()),
                                     *std::max_element(bending_scalars.begin(), bending_scalars.end())});

    std::vector<double> perp_scalars, III_scalars;
    for (int i = 0; i < mesh.nFaces(); i++) {
        perp_scalars.push_back(extra_energy_terms->compute_vector_perp_tangent_energy_perface(
            cur_pos, edge_dofs, mesh, rest_state.abars, i, nullptr, nullptr, false));
        III_scalars.push_back(extra_energy_terms->compute_thirdFundamentalForm_energy_perface(cur_pos, edge_dofs, mesh, rest_state.abars, i, nullptr, nullptr, false));
    }
    auto scalar_plot = surface_mesh->addFaceScalarQuantity("perp", perp_scalars);
    scalar_plot->setMapRange({*std::min_element(perp_scalars.begin(), perp_scalars.end()),
                                     *std::max_element(perp_scalars.begin(), perp_scalars.end())});
    auto III_plot = surface_mesh->addFaceScalarQuantity("III", III_scalars);
    III_plot->setMapRange({*std::min_element(III_scalars.begin(), III_scalars.end()),
                                     *std::max_element(III_scalars.begin(), III_scalars.end())});

    auto vec_quantity = pt_mesh->addVectorQuantity(model_name + " Edge Normals", face_edge_normals);
    vec_quantity->setEnabled(true);
}

struct InputArgs {
    double thickness = 0.01;
    double triangle_area = 0.002;
    int sff_model = 0;  // 0 for s1 and 2 for s2
    bool delaunlay_mesh = false;
    bool with_gui = false;
    int num_steps = 100;
};

int main(int argc, char* argv[]) {
    InputArgs args;
    CLI::App app{"Shell Energy Model"};
    app.add_option("-t,--thickness", args.thickness, "Thickness of the shell");
    app.add_option("-a,--triangle_area", args.triangle_area, "Relative triangle area of the mesh");
    app.add_option("-s,--sff_model", args.sff_model, "SFF model type");
    app.add_flag("-g,--with_gui", args.with_gui, "With GUI");
    app.add_flag("-d,--delaunay", args.delaunlay_mesh, "Cylinder with delaunay mesh");
    app.add_option("-n, --num_steps", args.num_steps, "Number of steps to run the simulation");
    CLI11_PARSE(app, argc, argv);

    double triangle_area = 0.002;
    if (args.triangle_area > 0) {
        triangle_area = args.triangle_area;
    }
    double thickness = 0.01;
    if (args.thickness > 0) {
        thickness = args.thickness;
    }
    SFFModelType sff_type = args.sff_model == 0 ? SFFModelType::kS1Sin : SFFModelType::kS2Sin;

    double cylinder_radius = 1;
    double cylinder_height = 1;

    // set up material parameters
    double young = 1e7;
    double poisson = 0;
    double shear = young / (2.0 * (1.0 + poisson));

    Eigen::MatrixXd origV;
    Eigen::MatrixXd rolledV;
    Eigen::MatrixXi F;

    double angle = M_PI;

    auto get_x_min_max = [](const Eigen::MatrixXd& V) {
        double xmin = std::numeric_limits<double>::infinity();
        double xmax = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < V.rows(); i++) {
            xmin = std::min(xmin, V(i, 0));
            xmax = std::max(xmax, V(i, 0));
        }
        return std::make_pair(xmin, xmax);
    };

    int cur_iter = 0;
    Eigen::VectorXd edge_dofs;
    LibShell::MonolayerRestState rest_state;
    Eigen::VectorXd edge_area;

    std::shared_ptr<LibShell::ExtraEnergyTermsBase> extra_energy_terms;
    std::shared_ptr<ShellEnergy> stvk_dir_energy_model;

    std::vector<Eigen::MatrixXd> cur_pos_frames(args.num_steps);
    std::vector<Eigen::VectorXd> cur_edge_dofs_frames(args.num_steps);

    bool reinitialization = true;

    bool is_paused = true;

    bool show_frames = false;
    bool show_current = true;
    LibShell::MeshConnectivity mesh;

    auto initialize_all = [&]() {
        mesh = LibShell::MeshConnectivity(F);
        std::tie(start_xmin, start_xmax) = get_x_min_max(rolledV);
        std::tie(left_side_vids, right_side_vids) = get_left_right_sides(origV, mesh);
        std::tie(stvk_dir_energy_model, extra_energy_terms) = initialization(mesh, rolledV, thickness, young, poisson, edge_dofs, rest_state, edge_area, sff_type);
        cur_pos_frames[0] = rolledV;
        cur_edge_dofs_frames[0] = edge_dofs;
        reinitialization = false;
    };

    if (args.with_gui) {
        makeCylinder(!args.delaunlay_mesh, cylinder_radius, cylinder_height,
                     triangle_area * angle * cylinder_radius * cylinder_height, origV, rolledV, F, angle);
        initialize_all();

        std::cout << "x min: " << start_xmin << " x max: " << start_xmax << std::endl;

        polyscope::init();
        surface_mesh = polyscope::registerSurfaceMesh("Current mesh", rolledV, F);
        surface_mesh->setEnabled(show_current);
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
        pt_mesh->setEnabled(show_current);
        update_rendering(surface_mesh, pt_mesh, stvk_dir_energy_model, extra_energy_terms, mesh, rest_state, rolledV, edge_dofs, sff_type);

        polyscope::state::userCallback = [&]() {
            if (ImGui::InputDouble("Triangle Area", &triangle_area)) {
                if (triangle_area <= 0) {
                    triangle_area = 0.02;
                }
                reinitialization = true;
            }

            if (ImGui::InputDouble("Thickness", &thickness)) {
                if (thickness <= 0) {
                    thickness = 0.001;
                }
                reinitialization = true;
            }

            if(ImGui::Combo("Bending Type", &args.sff_model, "S1 Sin\0S2 Sin\0")) {
                sff_type = args.sff_model == 0 ? SFFModelType::kS1Sin : SFFModelType::kS2Sin;
                initialize_all();
                cur_iter = 0;
                rolledV = cur_pos_frames[0];
                edge_dofs = cur_edge_dofs_frames[0];
                is_paused = true;
                update_rendering(surface_mesh, pt_mesh, stvk_dir_energy_model, extra_energy_terms, mesh, rest_state, rolledV, edge_dofs, sff_type);
            }

            ImGui::Checkbox("Include III", &is_include_III);

            if (ImGui::Button("Remake Cylinder", ImVec2(-1, 0))) {
                double actual_triangle_area = triangle_area * angle * cylinder_radius * cylinder_height;
                makeCylinder(angle, cylinder_radius, cylinder_height, actual_triangle_area, origV, rolledV, F, angle);
                reinitialization = true;
                initialize_all();
                surface_mesh = polyscope::registerSurfaceMesh("Current mesh", rolledV, F);
                surface_mesh->setEnabled(show_current);
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
                pt_mesh->setEnabled(show_current);
            }

            if(ImGui::Checkbox("Pause", &is_paused)) {
                if(is_paused) {
                    std::cout << "Pause the simulation" << std::endl;
                }
            }
            if (ImGui::Button("Reset", ImVec2(-1, 0))) {
                cur_iter = 0;
                rolledV = cur_pos_frames[0];
                edge_dofs = cur_edge_dofs_frames[0];
                is_paused = true;
                update_rendering(surface_mesh, pt_mesh, stvk_dir_energy_model, extra_energy_terms, mesh, rest_state, rolledV, edge_dofs, sff_type);
            }

            if (ImGui::Button("Squish", ImVec2(-1, 0)) || (!is_paused && cur_iter < args.num_steps)) {
                if(reinitialization) {
                    initialize_all();
                }
                is_paused = false;
                processFullModelOneStep(mesh, rest_state, edge_area, stvk_dir_energy_model, extra_energy_terms, young * thickness / args.thickness, rolledV,
                    edge_dofs, sff_type, cur_iter / (double)args.num_steps);
                cur_iter++;
                cur_pos_frames[cur_iter] = rolledV;
                cur_edge_dofs_frames[cur_iter] = edge_dofs;
                update_rendering(surface_mesh, pt_mesh, stvk_dir_energy_model, extra_energy_terms, mesh, rest_state, rolledV, edge_dofs, sff_type);

                polyscope::refresh();
                polyscope::requestRedraw();
            }

            if(ImGui::Checkbox("Show Frames", &show_frames)) {
            }
            if(ImGui::Checkbox("Show Current", &show_current)) {
                pt_mesh->setEnabled(show_current);
                surface_mesh->setEnabled(show_current);
            }
            // ** Add a slider to view different frames after simulation**
            if (cur_iter > 0) {
                ImGui::Text("Frame Viewer:");
                ImGui::SliderInt("Frame", &cur_frame_idx, 0, cur_iter);

                // If the user moves the slider, update the mesh visualization
                if (cur_frame_idx >= 0 && cur_frame_idx <= cur_iter) {
                    auto frame_surface_mesh = polyscope::registerSurfaceMesh("Frame mesh", cur_pos_frames[cur_frame_idx], F);
                    frame_surface_mesh->setEnabled(show_frames);
                    std::vector<Eigen::Vector3d> cur_face_edge_midpts = {};
                    for (int i = 0; i < mesh.nFaces(); i++) {
                        for (int j = 0; j < 3; j++) {
                            int eid = mesh.faceEdge(i, j);
                            Eigen::Vector3d midpt =
                                (cur_pos_frames[cur_frame_idx].row(mesh.edgeVertex(eid, 0)) + cur_pos_frames[cur_frame_idx].row(mesh.edgeVertex(eid, 1))) / 2.0;
                            cur_face_edge_midpts.push_back(midpt);
                        }
                    }
                    auto frame_pt_mesh = polyscope::registerPointCloud("Frame face edge midpoints", face_edge_midpts);
                    frame_pt_mesh->setEnabled(show_frames);
                    update_rendering(frame_surface_mesh, frame_pt_mesh, stvk_dir_energy_model, extra_energy_terms, mesh, rest_state, cur_pos_frames[cur_frame_idx], cur_edge_dofs_frames[cur_frame_idx], sff_type);
                    polyscope::requestRedraw();
                }
            }
        };

        polyscope::show();
    }

    return 0;
}
