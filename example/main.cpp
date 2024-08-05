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

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui.h>

#include <unordered_set>
#include <memory>

int numSteps;
double gradTol;
double fTol;
double xTol;

double thickness;
double poisson;
int matid;
int sffid;
int projType;

Eigen::MatrixXd curPos;
LibShell::MeshConnectivity mesh;

void repaint(igl::opengl::glfw::Viewer &viewer)
{
    viewer.data().clear();
    viewer.data().set_mesh(curPos, mesh.faces());    
}

void lameParameters(double &alpha, double &beta)
{
    double young = 1.0; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

template <class SFF>
void runSimulation(
    igl::opengl::glfw::Viewer &viewer, 
    const LibShell::MeshConnectivity &mesh, 
    const Eigen::MatrixXd &restPos,
    Eigen::MatrixXd &curPos, 
    const std::unordered_set<int> *fixedVerts,
    double thickness,
    double lameAlpha,
    double lameBeta,
    int matid,
    int proj_type)
{
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
    for (int i = 0; i < mesh.nFaces(); i++)
    {
        restState.bbars[i] = SFF::secondFundamentalForm(
            mesh, restPos, init_edgeDOFs, i, nullptr, nullptr);
        restState.bbars[i].setZero();
    }

    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);

    std::shared_ptr<LibShell::MaterialModel<SFF>> mat;
    switch (matid)
    {
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
    Eigen::VectorXd fixedDOFs(3 * curPos.rows());
    fixedDOFs.setZero();
    int nfree = 0;
    for (int i = 0; i < curPos.rows(); i++) {
        if (!fixedVerts || !fixedVerts->count(i)) {
            Pcoeffs.push_back({nfree, 3 * i, 1.0});
            Pcoeffs.push_back({nfree + 1, 3 * i + 1, 1.0});
            Pcoeffs.push_back({nfree + 2, 3 * i + 2, 1.0});
            nfree += 3;
        } else {
            fixedDOFs.segment<3>(3 * i) = curPos.row(i).transpose();
        }
    }
    for (int i = 0; i < nedges * nedgedofs; i++) {
        Pcoeffs.push_back(Eigen::Triplet<double>(nfree, 3 * curPos.rows() + i, 1.0));
        nfree++;
    }

    P.resize(nfree, 3 * curPos.rows() + nedges * nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

    int totalDOFs = 3 * curPos.rows() + nedges * nedgedofs;

    // project the current position
    auto pos_edgedofs_to_variable = [&](const Eigen::MatrixXd &pos, const Eigen::VectorXd &edgeDOFs) {
        Eigen::VectorXd var(nfree);
        int n = 0;
        for (int i = 0; i < pos.rows(); i++) {
            if (!fixedVerts || !fixedVerts->count(i)) {
                var.segment<3>(n) = pos.row(i).transpose();
                n += 3;
            }
        }
        var.tail(nedges * nedgedofs) = edgeDOFs;
        return var;
    };

    auto variable_to_pos_edgedofs = [&](const Eigen::VectorXd &var) {
        Eigen::MatrixXd pos(curPos.rows(), 3);
        int n = 0;
        for (int i = 0; i < curPos.rows(); i++) {
            if (!fixedVerts || !fixedVerts->count(i)) {
                pos.row(i) = var.segment<3>(n).transpose();
                n += 3;
            } else {
                pos.row(i) = fixedDOFs.segment<3>(3 * i).transpose();
            }
        }
        Eigen::VectorXd edgeDOFs = var.tail(nedges * nedgedofs);
        return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{pos, edgeDOFs};
    };

    // energy, gradient, and hessian
    auto obj_func = [&](const Eigen::VectorXd &var, Eigen::VectorXd *grad, Eigen::SparseMatrix<double> *hessian, bool psd_proj)
    {
        Eigen::MatrixXd pos;
        Eigen::VectorXd edgeDOFs;
        std::vector<Eigen::Triplet<double>> hessian_triplets;
        std::tie(pos, edgeDOFs) = variable_to_pos_edgedofs(var);

        double energy = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, pos, edgeDOFs, *mat, restState, psd_proj ? proj_type : 0, grad,
            hessian ? &hessian_triplets : nullptr);

        if(grad) {
            if(fixedVerts)
            {
                *grad = P * (*grad);
            }
        }

        if(hessian) {
            hessian->resize(totalDOFs, totalDOFs);
            hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
            if(fixedVerts)
            {
                *hessian = P * (*hessian) * P.transpose();
            }
        }

        return energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd &x, const Eigen::VectorXd &dir)
    {
        return 1.0;
    };

    Eigen::VectorXd x0 = pos_edgedofs_to_variable(curPos, init_edgeDOFs);

    OptSolver::TestFuncGradHessian(obj_func, x0);

    OptSolver::NewtonSolver(obj_func, find_max_step, x0, numSteps, gradTol, xTol, fTol, projType != 0, true);

    std::tie(curPos, init_edgeDOFs) = variable_to_pos_edgedofs(x0);

    viewer.data().set_vertices(curPos);

    repaint(viewer);
}

int main(int argc, char *argv[])
{    
    numSteps = 30;
    gradTol = 1e-6;
    fTol = 0;
    xTol = 0;

    // set up material parameters
    thickness = 1e-1;    
    poisson = 1.0 / 2.0;
    matid = 0;
    sffid = 0;
    projType = 0;

    // load mesh
    
    Eigen::MatrixXd origV;
    Eigen::MatrixXi F;

    std::vector<std::string> prefixes = { "./", "./example/", "../", "../example/" };

    bool found = false;
    for (auto& it : prefixes)
    {
        std::string fname = it + std::string("bunny.obj");
        if (igl::readOBJ(fname, origV, F))
        {
            found = true;
            break;            
        }
    }
    if (!found)
    {
        std::cerr << "Could not read example bunny.obj file" << std::endl;
        return -1;
    }
     
    // set up mesh connectivity
    mesh = LibShell::MeshConnectivity(F);

    // initial position
    curPos = origV;

    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&plugin);
    plugin.widgets.push_back(&menu);
   

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::Button("Reset", ImVec2(-1, 0)))
        {
            curPos = origV;
            repaint(viewer);
        }

        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Thickness", &thickness);
            ImGui::InputDouble("Poisson's Ration", &poisson);
            ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
            ImGui::Combo("Second Fundamental Form", &sffid, "TanTheta\0SinTheta\0Average\0\0");
        }

        
        if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Combo("Hessian Projection", &projType, "No Projection\0Max Zero\0Abs\0\0");
            ImGui::InputInt("Num Steps", &numSteps);
            ImGui::InputDouble("Gradient Tol", &gradTol);
            ImGui::InputDouble("Function Tol", &fTol);
            ImGui::InputDouble("Variable Tol", &xTol);

            if (ImGui::Button("Optimize Some Step", ImVec2(-1,0)))
            {
                double lameAlpha, lameBeta;
                lameParameters(lameAlpha, lameBeta);

                switch (sffid)
                {
                case 0:
                    runSimulation<LibShell::MidedgeAngleTanFormulation>(viewer, mesh, origV, curPos, nullptr, thickness,
                        lameAlpha, lameBeta, matid, projType);
                    break;
                case 1:
                    runSimulation<LibShell::MidedgeAngleSinFormulation>(viewer, mesh, origV, curPos, nullptr, thickness,
                        lameAlpha, lameBeta, matid, projType);
                    break;
                case 2:
                    runSimulation<LibShell::MidedgeAverageFormulation>(viewer, mesh, origV, curPos, nullptr, thickness,
                        lameAlpha, lameBeta, matid, projType);
                    break;
                default:
                    assert(false);
                }
            }
        }
    };

    viewer.data().set_face_based(false);
    repaint(viewer);
    viewer.launch();
}
