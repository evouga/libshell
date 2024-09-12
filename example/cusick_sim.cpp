#include "StaticSolve.h"
#include "../Optimization/include/NewtonDescent.h"

#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"

#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>

#include <unordered_set>
#include <memory>
#include <filesystem>

#include <CLI/CLI.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <chrono>

int num_steps;
double grad_tol;
double f_tol;
double x_tol;
bool is_swap = false;

double young;
double thickness;
double density;  // g/m^2
double poisson;
int matid;
int sffid;
int proj_type;
bool fixed_edge_dofs;

Eigen::MatrixXd cur_pos;
LibShell::MeshConnectivity mesh;

std::string output_folder = "";

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif  // !M_PI

void lame_parameters(double& alpha, double& beta) {
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

template <class SFF>
void run_simulation(const LibShell::MeshConnectivity& mesh,
                    const Eigen::MatrixXd& rest_pos,
                    Eigen::MatrixXd& cur_pos,
                    const std::unordered_set<int>* fixed_verts,
                    double thickness,
                    double lame_alpha,
                    double lame_beta,
                    int matid,
                    int proj_type,
                    bool is_fixed_edege_dofs,
                    const Eigen::Vector3d& gravity = Eigen::Vector3d::Zero()) {
    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd init_edge_DOFs;
    SFF::initializeExtraDOFs(init_edge_DOFs, mesh, cur_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state;

    // set uniform thicknesses
    rest_state.thicknesses.resize(mesh.nFaces(), thickness);

    // initialize first fundamental forms to those of input mesh
    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, rest_pos, rest_state.abars);

    // initialize second fundamental forms to those of input mesh
    rest_state.bbars.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++) {
        rest_state.bbars[i] = SFF::secondFundamentalForm(mesh, rest_pos, init_edge_DOFs, i, nullptr, nullptr);
        rest_state.bbars[i].setZero();
    }

    rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
    rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

    std::shared_ptr<LibShell::MaterialModel<SFF>> mat;
    switch (matid) {
        case 0:
            mat = std::make_shared<LibShell::NeoHookeanMaterial<SFF>>();
            break;
        case 1:
            mat = std::make_shared<LibShell::StVKMaterial<SFF>>();
            break;
        case 2:
            mat = std::make_shared<LibShell::TensionFieldStVKMaterial<SFF>>();
            break;
        default:
            assert(false);
    }

    // projection matrix
    Eigen::SparseMatrix<double> P;
    std::vector<Eigen::Triplet<double>> Pcoeffs;
    int nedges = mesh.nEdges();
    int nedgedofs = SFF::numExtraDOFs;
    // we only allow fixed vertices in the current implementation
    Eigen::VectorXd fixed_dofs(3 * cur_pos.rows() + nedges * nedgedofs);
    fixed_dofs.setZero();
    int nfree = 0;
    for (int i = 0; i < cur_pos.rows(); i++) {
        if (!fixed_verts || !fixed_verts->count(i)) {
            Pcoeffs.push_back({nfree, 3 * i, 1.0});
            Pcoeffs.push_back({nfree + 1, 3 * i + 1, 1.0});
            Pcoeffs.push_back({nfree + 2, 3 * i + 2, 1.0});
            nfree += 3;
        } else {
            fixed_dofs.segment<3>(3 * i) = cur_pos.row(i).transpose();
        }
    }
    for (int i = 0; i < nedges * nedgedofs; i++) {
        if (!is_fixed_edege_dofs) {
            Pcoeffs.push_back(Eigen::Triplet<double>(nfree, 3 * cur_pos.rows() + i, 1.0));
            nfree++;
        } else {
            fixed_dofs(3 * cur_pos.rows() + i) = init_edge_DOFs(i);
        }
    }

    P.resize(nfree, 3 * cur_pos.rows() + nedges * nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    int totalDOFs = 3 * cur_pos.rows() + nedges * nedgedofs;

    // project the current position
    auto pos_edgedofs_to_variable = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& edge_DOFs) {
        Eigen::VectorXd var(nfree);
        int n = 0;
        for (int i = 0; i < pos.rows(); i++) {
            if (!fixed_verts || !fixed_verts->count(i)) {
                var.segment<3>(n) = pos.row(i).transpose();
                n += 3;
            }
        }
        if (!is_fixed_edege_dofs) {
            var.tail(nedges * nedgedofs) = edge_DOFs;
        }
        return var;
    };

    auto variable_to_pos_edgedofs = [&](const Eigen::VectorXd& var) {
        Eigen::MatrixXd pos(cur_pos.rows(), 3);
        Eigen::VectorXd edge_DOFs(nedges * nedgedofs);
        int n = 0;
        for (int i = 0; i < cur_pos.rows(); i++) {
            if (!fixed_verts || !fixed_verts->count(i)) {
                pos.row(i) = var.segment<3>(n).transpose();
                n += 3;
            } else {
                pos.row(i) = fixed_dofs.segment<3>(3 * i).transpose();
            }
        }
        if (is_fixed_edege_dofs) {
            edge_DOFs = fixed_dofs.tail(nedges * nedgedofs);
        } else {
            edge_DOFs = var.tail(nedges * nedgedofs);
        }
        return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{pos, edge_DOFs};
    };

    Eigen::VectorXd vertex_masses = Eigen::VectorXd::Zero(cur_pos.rows());
    for (int i = 0; i < mesh.nFaces(); i++) {
        double area = 0.5 * rest_state.abars[i].determinant();
        for (int j = 0; j < 3; j++) {
            vertex_masses[mesh.faceVertex(i, j)] += area / 3.0;
        }
    }

    vertex_masses *= density / 1000;  // g/m^2 to kg/m^2

    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
                        bool psd_proj) {
        Eigen::MatrixXd pos;
        Eigen::VectorXd edge_DOFs;
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        std::tie(pos, edge_DOFs) = variable_to_pos_edgedofs(var);
        double energy =
            LibShell::ElasticShell<SFF>::elasticEnergy(mesh, pos, edge_DOFs, *mat, rest_state, psd_proj ? proj_type : 0,
                                                       grad, hessian ? &hessian_triplets : nullptr);

        // gravity
        if (gravity.norm() > 0) {
            for (int i = 0; i < pos.rows(); i++) {
                energy += -vertex_masses[i] * gravity.dot(pos.row(i).segment<3>(0));
                if (grad) {
                    grad->segment<3>(3 * i) -= gravity * vertex_masses[i];
                }
            }
        }

        if (grad) {
            if (fixed_verts || is_fixed_edege_dofs) {
                *grad = P * (*grad);
            }
        }

        if (hessian) {
            hessian->resize(totalDOFs, totalDOFs);
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            if (fixed_verts || is_fixed_edege_dofs) {
                *hessian = P * (*hessian) * P.transpose();
            }
        }

        return energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    Eigen::VectorXd x0 = pos_edgedofs_to_variable(cur_pos, init_edge_DOFs);

    if (output_folder != "") {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_folder + "/log.txt", true);
        spdlog::flush_every(std::chrono::seconds(1));

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto multi_sink_logger =
            std::make_shared<spdlog::logger>("Newton Solver", spdlog::sinks_init_list{file_sink, console_sink});

        spdlog::set_default_logger(multi_sink_logger);
    }
    OptSolver::NewtonSolver(obj_func, find_max_step, x0, num_steps, grad_tol, x_tol, f_tol, proj_type != 0, true,
                            is_swap);

    std::tie(cur_pos, init_edge_DOFs) = variable_to_pos_edgedofs(x0);
}

static void generate_anulus_mesh(
    double inner_radius, double outer_radius, double triangle_area, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
    double targetlength = 2.0 * std::sqrt(triangle_area * M_PI * (outer_radius * outer_radius - inner_radius * inner_radius)/ std::sqrt(3.0));
    int inner_n = std::max(1, int(2.0 * M_PI * inner_radius / targetlength));
    int outer_n = std::max(1, int(2.0 * M_PI * outer_radius / targetlength));

    Eigen::MatrixXd Vin(inner_n + outer_n, 2);
    Eigen::MatrixXi E(inner_n + outer_n, 2);

    for (int i = 0; i < inner_n; i++) {
        double theta = 2.0 * M_PI * i / inner_n;
        Vin(i, 0) = inner_radius * std::cos(theta);
        Vin(i, 1) = inner_radius * std::sin(theta);
        if (i < inner_n - 1) {
            E(i, 0) = i;
            E(i, 1) = (i + 1) % inner_n;
        } else {
            E(i, 0) = i;
            E(i, 1) = 0;
        }
    }

    for (int i = inner_n; i < inner_n + outer_n; i++) {
        double theta = 2.0 * M_PI * (i - inner_n) / outer_n;
        Vin(i, 0) = outer_radius * std::cos(theta);
        Vin(i, 1) = outer_radius * std::sin(theta);
        if (i < inner_n + outer_n - 1) {
            E(i, 0) = i;
            E(i, 1) = i + 1;
        } else {
            E(i, 0) = i;
            E(i, 1) = inner_n;
        }
    }

    Eigen::MatrixXd dummy_H(1, 2);
    dummy_H.row(0) << 0, 0;

    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;

    std::stringstream ss;
    ss << "a" << std::setprecision(30) << std::fixed << triangle_area << "qDY";
    std::cout << ss.str() << std::endl;
    igl::triangle::triangulate(Vin, E, dummy_H, ss.str(), V2, F2);

    F = std::move(F2);
    V.setZero(V2.rows(), 3);
    V.block(0, 0, V2.rows(), 2) = V2;
    
}

static void generate_plane_mesh(
    double width, double height, double triangle_area, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
    double targetlength = 2.0 * std::sqrt(triangle_area / std::sqrt(3.0));

    int W = std::max(1, int(width / targetlength));
    int H = std::max(1, int(height / targetlength));
    Eigen::MatrixXd Vin(2 * W + 2 * H, 2);
    Eigen::MatrixXi E(2 * W + 2 * H, 2);
    Eigen::MatrixXd dummy_H(0, 2);
    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;

    int vrow = 0;
    int erow = 0;
    // top boundary
    for (int i = 1; i < W; i++) {
        Vin(vrow, 0) = double(i) / double(W) * width;
        Vin(vrow, 1) = height;
        if (i > 1) {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // bottom boundary
    for (int i = 1; i < W; i++) {
        Vin(vrow, 0) = double(i) / double(W) * width;
        Vin(vrow, 1) = 0;
        if (i > 1) {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // left boundary
    for (int i = 0; i <= H; i++) {
        Vin(vrow, 0) = 0;
        Vin(vrow, 1) = double(i) / double(H) * height;
        if (i > 0) {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // right boundary
    for (int i = 0; i <= H; i++) {
        Vin(vrow, 0) = width;
        Vin(vrow, 1) = double(i) / double(H) * height;
        if (i > 0) {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // missing four edges
    E(erow, 0) = (W - 1) - 1;
    E(erow, 1) = 2 * (W - 1) + 2 * (H + 1) - 1;
    erow++;
    E(erow, 0) = 2 * (W - 1) + (H + 1);
    E(erow, 1) = 2 * (W - 1) - 1;
    erow++;
    E(erow, 0) = W - 1;
    E(erow, 1) = 2 * (W - 1);
    erow++;
    E(erow, 0) = 2 * (W - 1) + (H + 1) - 1;
    E(erow, 1) = 0;
    erow++;

    assert(vrow == 2 * H + 2 * W);
    assert(erow == 2 * H + 2 * W);
    std::stringstream ss;
    ss << "a" << std::setprecision(30) << std::fixed << triangle_area << "qDY";
    igl::triangle::triangulate(Vin, E, dummy_H, ss.str(), V2, F2);

    F = std::move(F2);
    V.setZero(V2.rows(), 3);
    V.block(0, 0, V2.rows(), 2) = V2;
}

int main(int argc, char* argv[]) {
    CLI::App app("Static Simulation for the Cusick Test");
    double triangle_area;
    double inner_radius;
    double outer_radius;
    bool no_gui = false;

    // trianlge area
    app.add_option("--triangle-area", triangle_area, "Plane relative triangle area")->default_val(1e-4);
    app.add_option("--inner-radius", inner_radius, "Inner radius of the anulus")->default_val(0.18);
    app.add_option("--outer-radius", outer_radius, "Outer radius of the anulus")->default_val(0.3);

    // optimization parameters
    app.add_option("--num-steps", num_steps, "Number of iteration")->default_val(1000);
    app.add_option("--grad-tol", grad_tol, "Gradient tolerance")->default_val(1e-6);
    app.add_option("--f-tol", f_tol, "Function tolerance")->default_val(0);
    app.add_option("--x-tol", x_tol, "Variable tolerance")->default_val(0);

    // material parameters
    app.add_option("--young", young, "Young's Modulus")->default_val(1e4);
    app.add_option("--thickness", thickness, "Thickness")->default_val(3e-4);
    app.add_option("--density", density, "Density (g/m^2)")->default_val(150);
    app.add_option("--poisson", poisson, "Poisson's Ratio")->default_val(0.3);
    app.add_option("--material", matid, "Material Model, 0: NeoHookean, 1: StVK")->default_val(1);
    app.add_option("--sff", sffid,
                   "Second Fundamental Form Formula, 0: midedge tan, 1: midedge sin, 2: midedge average")
        ->default_val(0);
    app.add_option("--projection", proj_type, "Hessian Projection Type, 0 : no projection, 1: max(H, 0), 2: Abs(H)")
        ->default_val(1);
    app.add_flag("--swap", is_swap, "Swap to Actual Hessian when close to optimum")->default_val(false);

    // fixed edge dofs
    app.add_flag("--fixed-edge-dofs", fixed_edge_dofs, "Fixed edge dofs")->default_val(false);

    app.add_option("ouput,-o,--output", output_folder, "Output folder");
    app.add_flag("--no-gui", no_gui, "Without gui")->default_val(false);
    CLI11_PARSE(app, argc, argv);

    // make output folder
    if (output_folder != "") {
        std::filesystem::create_directories(output_folder);
    }

    // generate mesh
    Eigen::MatrixXd orig_V, rest_V;
    Eigen::MatrixXi F;

    generate_anulus_mesh(inner_radius, outer_radius, triangle_area, rest_V, F);

    // gravity
    Eigen::Vector3d gravity(0, 0, -9.8);

    // // get the left and right boundary vertices
    std::unordered_set<int> fixed_verts;
    std::vector<std::vector<int>> boundary_loops;
    igl::boundary_loop(F, boundary_loops);

    for (auto& loop : boundary_loops) {
        bool inner_loop = false;
        for (auto& vid : loop) {
            if (std::abs(rest_V.row(vid).norm() - inner_radius) < std::abs(rest_V.row(vid).norm() - outer_radius)) {
                inner_loop = true;
                break;
            }
        }
        if (inner_loop) {
            for (auto& vid : loop) {
                fixed_verts.insert(vid);
            }
            break;
        }
    }

    // set up mesh connectivity
    mesh = LibShell::MeshConnectivity(F);

    // initial position
    cur_pos = rest_V;
    orig_V = cur_pos;

    if (no_gui) {
        double lame_alpha, lame_beta;
        lame_parameters(lame_alpha, lame_beta);

        switch (sffid) {
            case 0:
                run_simulation<LibShell::MidedgeAngleTanFormulation>(
                    mesh, rest_V, cur_pos, fixed_verts.empty() ? nullptr : &fixed_verts, thickness, lame_alpha,
                    lame_beta, matid, proj_type, fixed_edge_dofs, gravity);
                break;
            case 1:
                run_simulation<LibShell::MidedgeAngleSinFormulation>(
                    mesh, rest_V, cur_pos, fixed_verts.empty() ? nullptr : &fixed_verts, thickness, lame_alpha,
                    lame_beta, matid, proj_type, fixed_edge_dofs, gravity);
                break;
            case 2:
                run_simulation<LibShell::MidedgeAverageFormulation>(
                    mesh, rest_V, cur_pos, fixed_verts.empty() ? nullptr : &fixed_verts, thickness, lame_alpha,
                    lame_beta, matid, proj_type, fixed_edge_dofs, gravity);
                break;
            default:
                assert(false);
        }
        if (output_folder != "") {
            igl::writeOBJ(output_folder + "/rest.obj", rest_V, F);
            igl::writeOBJ(output_folder + "/orig.obj", orig_V, F);
            igl::writeOBJ(output_folder + "/deformed.obj", cur_pos, F);
        }
        return EXIT_SUCCESS;
    }

    polyscope::init();

    polyscope::view::setUpDir(polyscope::UpDir::ZUp);

    // Register a surface mesh structure
    auto surface_mesh = polyscope::registerSurfaceMesh("Rest mesh", rest_V, F);
    surface_mesh->setEnabled(false);

    auto cur_surface_mesh = polyscope::registerSurfaceMesh("Current mesh", cur_pos, F);

    std::vector<Eigen::RowVector3d> fixed_pos;
    for (auto& vid : fixed_verts) {
        fixed_pos.push_back(cur_pos.row(vid));
    }

    polyscope::registerPointCloud("Fixed Vertices", fixed_pos);
    
    polyscope::options::groundPlaneHeightFactor = 2 * outer_radius;

    if (output_folder != "") {
        igl::writeOBJ(output_folder + "/rest.obj", rest_V, F);
        igl::writeOBJ(output_folder + "/orig.obj", orig_V, F);
    }

    polyscope::state::userCallback = [&]() {
        if (ImGui::Button("Reset", ImVec2(-1, 0))) {
            cur_pos = orig_V;
            cur_surface_mesh->updateVertexPositions(cur_pos);
        }

        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputDouble("Young's Modulus", &young);
            ImGui::InputDouble("Thickness", &thickness);
            ImGui::InputDouble("Poisson's Ration", &poisson);
            ImGui::InputDouble("Density (g/m^2)", &density);
            ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
            ImGui::Combo("Second Fundamental Form", &sffid, "TanTheta\0SinTheta\0Average\0\0");
            ImGui::Checkbox("Fix Edge DOFs", &fixed_edge_dofs);
        }

        if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Combo("Hessian Projection", &proj_type, "No Projection\0Max Zero\0Abs\0\0");
            ImGui::InputInt("Num Steps", &num_steps);
            ImGui::InputDouble("Gradient Tol", &grad_tol);
            ImGui::InputDouble("Function Tol", &f_tol);
            ImGui::InputDouble("Variable Tol", &x_tol);
            ImGui::Checkbox("Swap to Actual Hessian when close to optimum", &is_swap);

            if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0))) {
                double lame_alpha, lame_beta;
                lame_parameters(lame_alpha, lame_beta);

                switch (sffid) {
                    case 0:
                        run_simulation<LibShell::MidedgeAngleTanFormulation>(
                            mesh, rest_V, cur_pos, fixed_verts.empty() ? nullptr : &fixed_verts, thickness, lame_alpha,
                            lame_beta, matid, proj_type, fixed_edge_dofs, gravity);
                        break;
                    case 1:
                        run_simulation<LibShell::MidedgeAngleSinFormulation>(
                            mesh, rest_V, cur_pos, fixed_verts.empty() ? nullptr : &fixed_verts, thickness, lame_alpha,
                            lame_beta, matid, proj_type, fixed_edge_dofs, gravity);
                        break;
                    case 2:
                        run_simulation<LibShell::MidedgeAverageFormulation>(
                            mesh, rest_V, cur_pos, fixed_verts.empty() ? nullptr : &fixed_verts, thickness, lame_alpha,
                            lame_beta, matid, proj_type, fixed_edge_dofs, gravity);
                        break;
                    default:
                        assert(false);
                }
                cur_surface_mesh->updateVertexPositions(cur_pos);
                if (output_folder != "") {
                    igl::writeOBJ(output_folder + "/deformed.obj", cur_pos, F);
                }
            }
        }
    };

    // View the point cloud and mesh we just registered in the 3D UI
    polyscope::show();
}
