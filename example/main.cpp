#include <igl/opengl/glfw/Viewer.h>
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "StaticSolve.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"

int numSteps;
double thickness;
double poisson;
int matid;
int sffid;

Eigen::MatrixXd curPos;
MeshConnectivity mesh;

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

int main(int argc, char *argv[])
{    
    numSteps = 30;

    // set up material parameters
    thickness = 1e-1;    
    poisson = 1.0 / 2.0;
    matid = 0;
    sffid = 0;

    // load mesh
    
    Eigen::MatrixXd origV;
    Eigen::MatrixXi F;
    if (!igl::readOBJ("bunny.obj", origV, F))
    {
        std::cerr << "Could not read example bunny.obj file" << std::endl;
        return -1;
    }
     
    // set up mesh connectivity
    mesh = MeshConnectivity(F);

    // initial position
    curPos = origV;

    // initialize first fundamental forms to those of input mesh
    std::vector<Eigen::Matrix2d> abar;
    firstFundamentalForms(mesh, curPos, abar);

    // initialize second fundamental forms to rest flat
    std::vector<Eigen::Matrix2d> bbar;
    bbar.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++)
        bbar[i].setZero();

    
    
    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
   

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
            ImGui::InputInt("Num Steps", &numSteps);
            if (ImGui::Button("Optimize Some Step", ImVec2(-1,0)))
            {
                Eigen::VectorXd thicknesses(mesh.nFaces());
                thicknesses.setConstant(thickness);
                MaterialModel *mat;
                double lameAlpha, lameBeta;
                lameParameters(lameAlpha, lameBeta);
                switch (matid)
                {
                case 0:
                    mat = new NeoHookeanMaterial(lameAlpha, lameBeta);
                    break;
                case 1:
                    mat = new StVKMaterial(lameAlpha, lameBeta);
                    break;
                default:
                    assert(false);
                }

                SecondFundamentalFormDiscretization *sff;
                switch (sffid)
                {
                case 0:
                    sff = new MidedgeAngleTanFormulation;
                    break;
                case 1:
                    sff = new MidedgeAngleSinFormulation;
                    break;
                case 2:
                    sff = new MidedgeAverageFormulation;
                    break;
                default:
                    assert(false);
                }

                // initialize default edge DOFs (edge director angles)
                Eigen::VectorXd edgeDOFs;
                sff->initializeExtraDOFs(edgeDOFs, mesh, curPos);

                double reg = 1e-6;
                for (int j = 1; j <= numSteps; j++)
                {                    
                    takeOneStep(mesh, curPos, edgeDOFs, *mat, thicknesses, abar, bbar, *sff, reg);
                    repaint(viewer);                
                }

                delete sff;
                delete mat;
            }
        }
    };

    viewer.data().set_face_based(false);
    repaint(viewer);
    viewer.launch();
}
