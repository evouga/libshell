#include "StaticSolve.h"
#include "../Optimization/include/NewtonDescent.h"

#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/MidedgeAngleCompressiveFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"

#include <polyscope/point_cloud.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <imgui.h>

#include <unordered_set>
#include <memory>

#include <CLI/CLI.hpp>

#include <igl/read_triangle_mesh.h>

double young = 1e6;
double poisson = 0.04;
double thickness = 1e-3;

int matid = 1;
int sffid = 2;

void lame_parameters(double &alpha, double &beta) {
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

template <class SFF>
void compute_bending_energy_density(polyscope::SurfaceMesh *rest_surface_mesh,
                                    polyscope::SurfaceMesh *current_surface_mesh,
                                    const LibShell::MeshConnectivity &mesh,
                                    const Eigen::MatrixXd &restPos,
                                    Eigen::MatrixXd &curPos,
                                    double thickness,
                                    double lameAlpha,
                                    double lameBeta,
                                    int matid) {
    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd init_edgeDOFs;
    SFF::initializeExtraDOFs(init_edgeDOFs, mesh, curPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);

    // initialize first fundamental forms to those of input mesh
    LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, restPos, restState.abars);

    // initialize second fundamental forms to those of input mesh
    restState.bbars.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++) {
        restState.bbars[i] = SFF::secondFundamentalForm(mesh, restPos, init_edgeDOFs, i, nullptr, nullptr);
    }

    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);

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

    std::vector<double> bending_energy_density = LibShell::ElasticShell<SFF>::elasticEnergyPerElement(
        mesh, curPos, init_edgeDOFs, *mat, restState, LibShell::ElasticShell<SFF>::ET_BENDING);

    rest_surface_mesh->addFaceScalarQuantity("Bending Energy Density", bending_energy_density);
    current_surface_mesh->addFaceScalarQuantity("Bending Energy Density", bending_energy_density);

    // compute the total bending energy
    double bending_energy = 0.0;
    for (auto &energy : bending_energy_density) {
        bending_energy += energy;
    }
    // std::accumulate(bending_energy_density.begin(), bending_energy_density.end(), bending_energy);
    std::cout << "Total bending energy: " << bending_energy << std::endl;
}

int main(int argc, char **argv) {
    struct {
        std::string input_rest_mesh;
        std::string input_current_mesh;
    } args;

    CLI::App app{argv[0]};
    app.add_option("--rest-mesh", args.input_rest_mesh, "Input Rest State")->required()->check(CLI::ExistingFile);
    app.add_option("--current-mesh", args.input_current_mesh, "Input current State")
        ->required()
        ->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv)

    Eigen::MatrixXd rest_V;
    Eigen::MatrixXi rest_F;
    igl::read_triangle_mesh(args.input_rest_mesh, rest_V, rest_F);

    Eigen::MatrixXd cur_V;
    Eigen::MatrixXi cur_F;
    igl::read_triangle_mesh(args.input_current_mesh, cur_V, cur_F);

    LibShell::MeshConnectivity mesh(rest_F);
    assert((rest_F - cur_F).norm() == 0);

    // Initialize polyscope
    polyscope::init();

    // Register a surface mesh structure
    auto rest_surf = polyscope::registerSurfaceMesh("Rest mesh", rest_V, rest_F);
    auto cur_surf = polyscope::registerSurfaceMesh("Current mesh", cur_V, cur_F);

    polyscope::state::userCallback = [&]() {
        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::InputDouble("Young's Modulus", &young);
            ImGui::InputDouble("Thickness", &thickness);
            ImGui::InputDouble("Poisson's Ration", &poisson);
            ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
            ImGui::Combo("Second Fundamental Form", &sffid, "TanTheta\0SinTheta\0Average\0Compressive\0\0");
        }

        if (ImGui::Button("Draw Bending Density", ImVec2(-1, 0))) {
            double lame_alpha, lame_beta;
            lame_parameters(lame_alpha, lame_beta);

            switch (sffid) {
                case 0:
                    compute_bending_energy_density<LibShell::MidedgeAngleTanFormulation>(
                        rest_surf, cur_surf, mesh, rest_V, cur_V, thickness, lame_alpha, lame_beta, matid);
                    break;
                case 1:
                    compute_bending_energy_density<LibShell::MidedgeAngleSinFormulation>(
                        rest_surf, cur_surf, mesh, rest_V, cur_V, thickness, lame_alpha, lame_beta, matid);
                    break;
                case 2:
                    compute_bending_energy_density<LibShell::MidedgeAverageFormulation>(
                        rest_surf, cur_surf, mesh, rest_V, cur_V, thickness, lame_alpha, lame_beta, matid);
                    break;
                case 3:
                    compute_bending_energy_density<LibShell::MidedgeAngleCompressiveFormulation>(
                        rest_surf, cur_surf, mesh, rest_V, cur_V, thickness, lame_alpha, lame_beta, matid);
                    break;
                default:
                    assert(false);
            }
        }
    };

    // View the point cloud and mesh we just registered in the 3D UI
    polyscope::show();

    return 0;
}
