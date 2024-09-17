#include "../optimization/include/NewtonDescent.h"

#include "../include/ElasticShell.h"
#include "../include/MeshConnectivity.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/types.h"

#include <polyscope/surface_mesh.h>

#include <igl/readOBJ.h>

#include <filesystem>
#include <memory>
#include <unordered_set>

#include <chrono>

int num_steps = 100;
double grad_tol = 1e-6;
double f_tol = 0;
double x_tol = 0;
bool is_swap = true;

double young = 1;
double thickness = 1e-1;
double poisson = 0.5;
int matid = 0;
int sffid = 0;
LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kMaxZero;

Eigen::MatrixXd cur_pos;
LibShell::MeshConnectivity mesh;


void lame_parameters(double &alpha, double &beta) {
  alpha = young * poisson / (1.0 - poisson * poisson);
  beta = young / 2.0 / (1.0 + poisson);
}


template <class SFF>
void run_simulation(const LibShell::MeshConnectivity &mesh,
                    const Eigen::MatrixXd &rest_pos, Eigen::MatrixXd &cur_pos,
                    const std::unordered_set<int> *fixed_verts,
                    double thickness, double lame_alpha, double lame_beta,
                    int matid, LibShell::HessianProjectType proj_type) {
  // initialize default edge DOFs (edge director angles)
  Eigen::VectorXd init_edge_DOFs;
  SFF::initializeExtraDOFs(init_edge_DOFs, mesh, cur_pos);

  // initialize the rest geometry of the shell
  LibShell::MonolayerRestState rest_state;

  // set uniform thicknesses
  rest_state.thicknesses.resize(mesh.nFaces(), thickness);

  // initialize first fundamental forms to those of input mesh
  LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, rest_pos,
                                                     rest_state.abars);

  // initialize second fundamental forms to those of input mesh
  rest_state.bbars.resize(mesh.nFaces());
  for (int i = 0; i < mesh.nFaces(); i++) {
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
  Eigen::VectorXd fixed_dofs(3 * cur_pos.rows());
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
    Pcoeffs.push_back(
        Eigen::Triplet<double>(nfree, 3 * cur_pos.rows() + i, 1.0));
    nfree++;
  }

  P.resize(nfree, 3 * cur_pos.rows() + nedges * nedgedofs);
  P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

  int totalDOFs = 3 * cur_pos.rows() + nedges * nedgedofs;

  // project the current position
  auto pos_edgedofs_to_variable = [&](const Eigen::MatrixXd &pos,
                                      const Eigen::VectorXd &edge_DOFs) {
    Eigen::VectorXd var(nfree);
    int n = 0;
    for (int i = 0; i < pos.rows(); i++) {
      if (!fixed_verts || !fixed_verts->count(i)) {
        var.segment<3>(n) = pos.row(i).transpose();
        n += 3;
      }
    }
    var.tail(nedges * nedgedofs) = edge_DOFs;
    return var;
  };

  auto variable_to_pos_edgedofs = [&](const Eigen::VectorXd &var) {
    Eigen::MatrixXd pos(cur_pos.rows(), 3);
    int n = 0;
    for (int i = 0; i < cur_pos.rows(); i++) {
      if (!fixed_verts || !fixed_verts->count(i)) {
        pos.row(i) = var.segment<3>(n).transpose();
        n += 3;
      } else {
        pos.row(i) = fixed_dofs.segment<3>(3 * i).transpose();
      }
    }
    Eigen::VectorXd edge_DOFs = var.tail(nedges * nedgedofs);
    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{pos, edge_DOFs};
  };

  // energy, gradient, and hessian
  auto obj_func = [&](const Eigen::VectorXd &var, Eigen::VectorXd *grad,
                      Eigen::SparseMatrix<double> *hessian, bool psd_proj) {
    Eigen::MatrixXd pos;
    Eigen::VectorXd edge_DOFs;
    std::vector<Eigen::Triplet<double>> hessian_triplets;
    std::tie(pos, edge_DOFs) = variable_to_pos_edgedofs(var);

    double energy = LibShell::ElasticShell<SFF>::elasticEnergy(
        mesh, pos, edge_DOFs, *mat, rest_state, grad,
        hessian ? &hessian_triplets : nullptr, psd_proj ? proj_type : LibShell::HessianProjectType::kNone);

    if (grad) {
      if (fixed_verts) {
        *grad = P * (*grad);
      }
    }

    if (hessian) {
      hessian->resize(totalDOFs, totalDOFs);
      hessian->setFromTriplets(hessian_triplets.begin(),
                               hessian_triplets.end());
      if (fixed_verts) {
        *hessian = P * (*hessian) * P.transpose();
      }
    }

    return energy;
  };

  auto find_max_step = [&](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &dir) { return 1.0; };

  Eigen::VectorXd x0 = pos_edgedofs_to_variable(cur_pos, init_edge_DOFs);
  OptSolver::TestFuncGradHessian(obj_func, x0);

  OptSolver::NewtonSolver(obj_func, find_max_step, x0, num_steps, grad_tol,
                          x_tol, f_tol, proj_type != LibShell::HessianProjectType::kNone, true, is_swap);

  std::tie(cur_pos, init_edge_DOFs) = variable_to_pos_edgedofs(x0);
}


int main(int argc, char *argv[]) {
  // generate mesh
  Eigen::MatrixXd orig_V, rest_V;
  Eigen::MatrixXi F;

  std::vector<std::string> prefixes = {"./", "./example/", "../",
                                       "../example/"};

  bool found = false;
  for (auto &it : prefixes) {
    std::string fname = it + std::string("bunny.obj");
    if (igl::readOBJ(fname, orig_V, F)) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::cerr << "Could not read example bunny.obj file" << std::endl;
    return -1;
  }

  rest_V = orig_V;
  // set up mesh connectivity
  mesh = LibShell::MeshConnectivity(F);
  // initial position
  cur_pos = rest_V;

  polyscope::init();

  // Register a surface mesh structure
  auto surface_mesh = polyscope::registerSurfaceMesh("Rest mesh", rest_V, F);
  surface_mesh->setEnabled(false);

  auto cur_surface_mesh =
      polyscope::registerSurfaceMesh("Current mesh", cur_pos, F);

  polyscope::state::userCallback = [&]() {
    if (ImGui::Button("Reset", ImVec2(-1, 0))) {
      cur_pos = orig_V;
      cur_surface_mesh->updateVertexPositions(cur_pos);
    }

    if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputDouble("Young's Modulus", &young);
      ImGui::InputDouble("Thickness", &thickness);
      ImGui::InputDouble("Poisson's Ration", &poisson);
      ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
      ImGui::Combo("Second Fundamental Form", &sffid,
                   "TanTheta\0SinTheta\0Average\0\0");
    }

    if (ImGui::CollapsingHeader("Optimization",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Combo("Hessian Projection", (int*)(&proj_type),
                   "No Projection\0Max Zero\0Abs\0\0");
      ImGui::InputInt("Num Steps", &num_steps);
      ImGui::InputDouble("Gradient Tol", &grad_tol);
      ImGui::InputDouble("Function Tol", &f_tol);
      ImGui::InputDouble("Variable Tol", &x_tol);
      ImGui::Checkbox("Swap to Actual Hessian Near Optimum", &is_swap);

      if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0))) {
        double lame_alpha, lame_beta;
        lame_parameters(lame_alpha, lame_beta);

        switch (sffid) {
        case 0:
          run_simulation<LibShell::MidedgeAngleTanFormulation>(
              mesh, rest_V, cur_pos, nullptr, thickness, lame_alpha,
              lame_beta, matid, proj_type);
          break;
        case 1:
          run_simulation<LibShell::MidedgeAngleSinFormulation>(
              mesh, rest_V, cur_pos, nullptr, thickness, lame_alpha, lame_beta,
              matid, proj_type);
          break;
        case 2:
          run_simulation<LibShell::MidedgeAverageFormulation>(
              mesh, rest_V, cur_pos, nullptr, thickness, lame_alpha, lame_beta,
              matid, proj_type);
          break;
        default:
          assert(false);
        }
        cur_surface_mesh->updateVertexPositions(cur_pos);
      }
    }
  };

  // View the point cloud and mesh we just registered in the 3D UI
  polyscope::show();
}
