#include <igl/opengl/glfw/Viewer.h>
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "StaticSolve.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include "../include/MidedgeAngleTanFormulation.h"

int numSteps;
double thickness;

Eigen::MatrixXd curPos;
Eigen::VectorXd edgeDOFs;
MeshConnectivity mesh;

void repaint(igl::opengl::glfw::Viewer &viewer)
{
    viewer.data().clear();
    viewer.data().set_mesh(curPos, mesh.faces());    
}

int main(int argc, char *argv[])
{    
    numSteps = 30;
    thickness = 1e-1;    

    MidedgeAngleTanFormulation sff;
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ("bunny.obj", V, F))
    {
        std::cerr << "Could not read example bunny.obj file" << std::endl;
        return -1;
    }
     
    // set up mesh connectivity
    mesh = MeshConnectivity(F);

    // initial position
    curPos = V;

    // initial edge DOFs
    sff.initializeExtraDOFs(edgeDOFs, mesh, curPos);

    // initialize first fundamental forms to those of input mesh
    std::vector<Eigen::Matrix2d> abar;
    firstFundamentalForms(mesh, curPos, abar);

    // initialize second fundamental forms to rest flat
    std::vector<Eigen::Matrix2d> bbar;
    bbar.resize(mesh.nFaces());
    for (int i = 0; i < mesh.nFaces(); i++)
        bbar[i].setZero();

    // set up material parameters
    double lameAlpha = 1.0;
    double lameBeta = 1.0;
    
    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Thickness", &thickness);
        }

        
        if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen))
        {            
            ImGui::InputInt("Num Steps", &numSteps);
            if (ImGui::Button("Optimize Some Step", ImVec2(-1,0)))
            {
                double reg = 1e-6;
                for (int j = 1; j <= numSteps; j++)
                {
                    takeOneStep(mesh, curPos, edgeDOFs, lameAlpha, lameBeta, thickness, abar, bbar, sff, reg);
                    repaint(viewer);                
                }
            }
        }
    };


    viewer.data().set_face_based(false);
    repaint(viewer);
    viewer.launch();
}
