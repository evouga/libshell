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

#include "../Optimization/include/NewtonDescent.h"
#include "igl/null.h"

#include "make_geometric_shapes/HalfCylinder.h"
#include "make_geometric_shapes/Sphere.h"

#include <polyscope/surface_vector_quantity.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <igl/readOBJ.h>
#include <igl/writePLY.h>
#include <igl/principal_curvature.h>
#include <set>
#include <vector>




double cokeRadius;
double cokeHeight;
double sphereRadius;

double thickness;
double poisson;

double triangleArea;

double QBEnergy;

enum class MeshType
{
    MT_CYLINDER_IRREGULAR,
    MT_CYLINDER_REGULAR,
    MT_SPHERE
};

MeshType curMeshType;

void lameParameters(double& alpha, double& beta)
{
    double young = 1.0 / thickness; // doesn't matter for static solves
    alpha = young * poisson / (1.0 - poisson * poisson);
    beta = young / 2.0 / (1.0 + poisson);
}

struct Energies
{
    double exact;                   // the theoretic II
    double quadratic_bending;       // quadratic bending
    double stvk;                    // stvk bending with fixed edge normal (average adjacent face normals)
    double stvk_s1_dir_sin;         // stvk bending with s1 direction, and sin formulation
    double stvk_s1_dir_tan;         // stvk bending with s1 direction, and tan formulation
    double stvk_s2_dir_sin;         // stvk bending with s2 direction, and sin formulation
    double stvk_s2_dir_tan;         // stvk bending with s2 direction, and tan formulation
};

double edgeDOFPenaltyEnergy(const Eigen::VectorXd& edgeDOFs, const Eigen::VectorXd&edge_area, double penaltyScale, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double> >* hessian)
{
    int nedges = edge_area.size();
    if (deriv) {
        deriv->resize(edgeDOFs.size());
        deriv->setZero();
    }

    if (hessian) {
        hessian->clear();
    }

    if(edgeDOFs.size() == nedges || edgeDOFs.size() % nedges != 0)
        return 0.0;
    int ndof_per_edge = edgeDOFs.size() / nedges;

    double penalty = 0.0;
    for (int i = 0; i < nedges; i++) {
        penalty += (edgeDOFs(i * ndof_per_edge + 1) - 1) * (edgeDOFs(i * ndof_per_edge + 1) - 1) * penaltyScale * edge_area[i];
        penalty += (edgeDOFs(i * ndof_per_edge + 2) - 1) * (edgeDOFs(i * ndof_per_edge + 2) - 1) * penaltyScale * edge_area[i];

        if(deriv) {
            (*deriv)(i * ndof_per_edge + 1) += 2 * (edgeDOFs(i * ndof_per_edge + 1) - 1) * penaltyScale * edge_area[i];
            (*deriv)(i * ndof_per_edge + 2) += 2 * (edgeDOFs(i * ndof_per_edge + 2) - 1) * penaltyScale * edge_area[i];
        }

        if(hessian) {
            hessian->push_back({i * ndof_per_edge + 1, i * ndof_per_edge + 1, 2 * penaltyScale * edge_area[i]});
            hessian->push_back({i * ndof_per_edge + 2, i * ndof_per_edge + 2, 2 * penaltyScale * edge_area[i]});
        }
    } 
    return penalty;
}


void optimizeEdgeDOFs(ShellEnergy& energy, const Eigen::MatrixXd& cur_pos, const Eigen::VectorXd& edge_area,  Eigen::VectorXd& edgeDOFs)
{
    double tol = 1e-5;
    int nposdofs = cur_pos.rows() * 3;
    int nedgedofs = edgeDOFs.size();

    std::vector<Eigen::Triplet<double> > Pcoeffs;
    std::vector<Eigen::Triplet<double> > Icoeffs;
    for (int i = 0; i < nedgedofs; i++)
    {
        Pcoeffs.push_back({ i, nposdofs + i, 1.0 });
        Icoeffs.push_back({ i, i, 1.0 });
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
        
        double elastic_energy = energy.elasticEnergy(cur_pos, var, true, grad, hessian ? &hessian_triplets : nullptr,
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
        return total_energy;
    };

    auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

    std::cout << "At beginning, elastic energy: " << energy.elasticEnergy(cur_pos, edgeDOFs, true, NULL, NULL)
              << std::endl;

    OptSolver::NewtonSolver(obj_func, find_max_step, edgeDOFs, 1000, 1e-5, 1e-15, 1e-15, true, true, true);

    std::cout << "At the end, elastic energy: " << energy.elasticEnergy(cur_pos, edgeDOFs, true, NULL, NULL)
              << std::endl;


    /*double reg = 1e-6;
    while (true)
    {
        std::vector<Eigen::Triplet<double> > Hcoeffs;
        Eigen::VectorXd F;
        double origEnergy = energy.elasticEnergy(cur_pos, edgeDOFs, true, &F, &Hcoeffs);
        
        std::vector<Eigen::Triplet<double>> PenaltyCoeffs;
        Eigen::VectorXd PenaltyDeriv;
        double penalty = edgeDOFPenaltyEnergy(edgeDOFs, edge_area, penaltyScale, &PenaltyDeriv, &PenaltyCoeffs);
        origEnergy += penalty;
        F.segment(nposdofs, nedgedofs) += PenaltyDeriv;
        for (auto& it : PenaltyCoeffs)
            Hcoeffs.push_back({it.row() + nposdofs, it.col() + nposdofs, it.value()});

        Eigen::VectorXd PF = P * F;
        std::cout << "Force resid now: " << PF.norm() << ", energy: " << origEnergy - penalty << ", penalty: " << penalty << ", reg: " << reg << std::endl;
        if (PF.norm() < tol) return;

        Eigen::SparseMatrix<double> H(nposdofs + nedgedofs, nposdofs + nedgedofs);
        H.setFromTriplets(Hcoeffs.begin(), Hcoeffs.end());
        Eigen::SparseMatrix<double> PHPT = P * H * PT;
        Eigen::SparseMatrix<double> M = PHPT + reg * I;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(M);
        Eigen::VectorXd update = solver.solve(-PF);
        if (solver.info() != Eigen::Success) {
            std::cout << "Solve failed" << std::endl;
            reg *= 2.0;
            continue;
        }
        Eigen::VectorXd newedgeDOFs = edgeDOFs + update;
        double newenergy = energy.elasticEnergy(cur_pos, newedgeDOFs, true, NULL, NULL);
        if (newenergy > origEnergy)
        {
            std::cout << "Not a descent step, " << origEnergy << " -> " << newenergy << std::endl;
            reg *= 2.0;
            continue;
        }
        edgeDOFs = newedgeDOFs;
        reg *= 0.5;
    }*/
}


Energies measureCylinderEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& rest_pos,
    const Eigen::MatrixXd& cur_pos,
    double thickness,
    double lame_alpha,
    double lame_beta,
    double cur_radius,
    double cur_height,
    Eigen::MatrixXd &nhForces,
    Eigen::MatrixXd &qbForces) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edge_dofs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edge_dofs, mesh,
                                                             rest_pos);

    Eigen::VectorXd zero_s1_dir_edge_dofs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zero_s1_dir_edge_dofs, mesh,
                                                              rest_pos);

    Eigen::VectorXd half_pi_zero_s2_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(half_pi_zero_s2_dir_edge_dofs, mesh,
                                                              rest_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state, s1_dir_rest_state, s2_dir_rest_state;

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

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        firstFundamentalForms(mesh, rest_pos, rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        secondFundamentalForms(mesh, rest_pos, edge_dofs, rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        firstFundamentalForms(mesh, rest_pos, s1_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        secondFundamentalForms(mesh, rest_pos, zero_s1_dir_edge_dofs, s1_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::
        firstFundamentalForms(mesh, rest_pos, s2_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::
        secondFundamentalForms(mesh, rest_pos, half_pi_zero_s2_dir_edge_dofs, s2_dir_rest_state.bbars);

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<
        LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh,
                                                                     cur_pos,
                                                                     edge_dofs,
                                                                     cur_bs);

    Eigen::VectorXd rest_edge_dofs = edge_dofs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, rest_pos,
                                              rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);
    result.stvk = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);

    StVKS1DirectorSinShellEnergy stvk_s1_dir_sin_energy_model(mesh, s1_dir_rest_state);
    StVKS1DirectorTanShellEnergy stvk_s1_dir_tan_energy_model(mesh, s1_dir_rest_state);
    StVKS2DirectorSinShellEnergy stvk_s2_dir_sin_energy_model(mesh, s2_dir_rest_state);
    StVKS2DirectorTanShellEnergy stvk_s2_dir_tan_energy_model(mesh, s2_dir_rest_state);

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if(fid != -1) {
                edge_area[i] += std::sqrt(rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        } 
    }

    Eigen::VectorXd s1_dir_sin_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_sin_energy_model, cur_pos, edge_area, s1_dir_sin_edge_dofs);
    result.stvk_s1_dir_sin = stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s1_dir_tan_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_tan_energy_model, cur_pos, edge_area, s1_dir_tan_edge_dofs);
    result.stvk_s1_dir_tan = stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s2_dir_sin_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_sin_energy_model, cur_pos, edge_area, s2_dir_sin_edge_dofs);
    result.stvk_s2_dir_sin = stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s2_dir_tan_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_tan_energy_model, cur_pos, edge_area, s2_dir_tan_edge_dofs);
    result.stvk_s2_dir_tan = stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, true, nullptr, nullptr);

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
    double svnorm =
        lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = PI * cur_radius * cur_height;

    result.exact = svnorm * coeff * area;

    return result;
}

Energies measureSphereEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& cur_pos,
    double thickness,
    double lame_alpha,
    double lame_beta,
    double radius) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edge_dofs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edge_dofs, mesh,
                                                             cur_pos);

    Eigen::VectorXd zero_s1_dir_edge_dofs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zero_s1_dir_edge_dofs, mesh,
                                                              cur_pos);

    Eigen::VectorXd half_pi_zero_s2_dir_edge_dofs;
    LibShell::MidedgeAngleGeneralTanFormulation::initializeExtraDOFs(half_pi_zero_s2_dir_edge_dofs, mesh,
                                                              cur_pos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState rest_state, s1_dir_rest_state, s2_dir_rest_state;

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

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        firstFundamentalForms(mesh, cur_pos, rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        secondFundamentalForms(mesh, cur_pos, edge_dofs, rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        firstFundamentalForms(mesh, cur_pos, s1_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        secondFundamentalForms(mesh, cur_pos, zero_s1_dir_edge_dofs, s1_dir_rest_state.bbars);

    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::
        firstFundamentalForms(mesh, cur_pos, s2_dir_rest_state.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::
        secondFundamentalForms(mesh, cur_pos, half_pi_zero_s2_dir_edge_dofs, s2_dir_rest_state.bbars);

    for(int i = 0; i < mesh.nFaces(); i++) {
        rest_state.bbars[i].setZero();
        s1_dir_rest_state.bbars[i].setZero();
        s2_dir_rest_state.bbars[i].setZero();
    }

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<
        LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh,
                                                                     cur_pos,
                                                                     edge_dofs,
                                                                     cur_bs);

    Eigen::VectorXd rest_edge_dofs = edge_dofs;
    QuadraticBendingShellEnergy qbenergyModel(mesh, rest_state, cur_pos,
                                              rest_edge_dofs);

    StVKShellEnergy stvk_energy_model(mesh, rest_state);

    result.quadratic_bending = qbenergyModel.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);
    result.stvk = stvk_energy_model.elasticEnergy(cur_pos, rest_edge_dofs, true, nullptr, nullptr);

    StVKS1DirectorSinShellEnergy stvk_s1_dir_sin_energy_model(mesh, s1_dir_rest_state);
    StVKS1DirectorTanShellEnergy stvk_s1_dir_tan_energy_model(mesh, s1_dir_rest_state);
    StVKS2DirectorSinShellEnergy stvk_s2_dir_sin_energy_model(mesh, s2_dir_rest_state);
    StVKS2DirectorTanShellEnergy stvk_s2_dir_tan_energy_model(mesh, s2_dir_rest_state);

    Eigen::VectorXd edge_area = Eigen::VectorXd::Zero(mesh.nEdges());

    for (int i = 0; i < mesh.nEdges(); i++) {
        for (int j = 0; j < 2; j++) {
            int fid = mesh.edgeFace(i, j);
            if(fid != -1) {
                edge_area[i] += std::sqrt(rest_state.abars[fid].determinant()) / 2.0 / 3.0;
            }
        }
    }

    Eigen::VectorXd s2_dir_sin_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S2 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_sin_energy_model, cur_pos, edge_area, s2_dir_sin_edge_dofs);
    result.stvk_s2_dir_sin = stvk_s2_dir_sin_energy_model.elasticEnergy(cur_pos, s2_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s1_dir_sin_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Sin) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_sin_energy_model, cur_pos, edge_area, s1_dir_sin_edge_dofs);
    result.stvk_s1_dir_sin = stvk_s1_dir_sin_energy_model.elasticEnergy(cur_pos, s1_dir_sin_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s1_dir_tan_edge_dofs = zero_s1_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S1 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s1_dir_tan_energy_model, cur_pos, edge_area, s1_dir_tan_edge_dofs);
    result.stvk_s1_dir_tan = stvk_s1_dir_tan_energy_model.elasticEnergy(cur_pos, s1_dir_tan_edge_dofs, true, nullptr, nullptr);

    Eigen::VectorXd s2_dir_tan_edge_dofs = half_pi_zero_s2_dir_edge_dofs;
    std::cout << "============= Optimizing edge direction (S2 Tan) =========== " << std::endl;
    optimizeEdgeDOFs(stvk_s2_dir_tan_energy_model, cur_pos, edge_area, s2_dir_tan_edge_dofs);
    result.stvk_s2_dir_tan = stvk_s2_dir_tan_energy_model.elasticEnergy(cur_pos, s2_dir_tan_edge_dofs, true, nullptr, nullptr);

    // ground truth energy
    Eigen::Matrix2d abar;
    abar.setIdentity();

    Eigen::Matrix2d b;
    b << 1.0 / radius, 0, 0, 1.0 / radius;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm =
        lame_alpha / 2.0 * M.trace() * M.trace() + lame_beta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = 4.0 * PI * radius * radius;

    result.exact = svnorm * coeff * area;

    return result;
}

int main(int argc, char* argv[])
{
    cokeRadius = 0.0325;
    cokeHeight = 0.122;
    sphereRadius = 0.05;

    // triangleArea = 0.0000001;
    triangleArea = 0.00001;

    // curMeshType = MeshType::MT_CYLINDER_REGULAR;
    curMeshType = MeshType::MT_CYLINDER_IRREGULAR;
    // curMeshType = MeshType::MT_SPHERE;
    
    Energies curenergies;
    QBEnergy = 0;

    Eigen::MatrixXd nhForces;
    Eigen::MatrixXd qbForces;


    // set up material parameters
    thickness = 1.0;// 0.00010;
    poisson = 1.0 / 2.0;

    // load mesh

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
    log << std::setw(15) << "#V" << "| " << std::setw(15) << "exact energy" << "| "
        << std::setw(25) << "Quadratic bending energy" << "| "
        << std::setw(20) << "StVK energy" << "| "
        << std::setw(20) << "StVK_S1_sin energy" << "| "
        << std::setw(20) << "StVK_S1_tan energy" << "| "
        << std::setw(20) << "StVK_S2_sin energy" << "| "
        << std::setw(20) << "StVK_S2_tan energy" << std::endl;
    if (curMeshType == MeshType::MT_SPHERE)
    {
        makeSphere(sphereRadius, triangleArea, origV, F);
        LibShell::MeshConnectivity mesh(F);
        for (int step = 0; step < steps; step++)
        {
            std::stringstream ss;
            ss << "sphere_ " << step << ".ply";
            igl::writePLY(ss.str(), origV, F);

            curenergies = measureSphereEnergy(mesh, origV, thickness, lame_alpha, lame_beta, sphereRadius);
            log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| "
                << std::setw(25) << curenergies.quadratic_bending << "| "
                << std::setw(20) << curenergies.stvk << "| "
                << std::setw(20) << curenergies.stvk_s1_dir_sin << "| "
                << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
                << std::setw(20) << curenergies.stvk_s2_dir_sin << "| "
                << std::setw(20) << curenergies.stvk_s2_dir_tan << std::endl;
            triangleArea *= multiplier;
            makeSphere(sphereRadius, triangleArea, origV, F);
            mesh = LibShell::MeshConnectivity(F);            
        }
    }
    else
    {
        makeHalfCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);
        LibShell::MeshConnectivity mesh(F);
        for (int step = 0; step < steps; step++)
        {
            curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, lame_alpha, lame_beta, cur_radius, cur_height, nhForces, qbForces);

            log << std::setw(15) << origV.rows() << "| " << std::setw(15) << curenergies.exact << "| "
                << std::setw(25) << curenergies.quadratic_bending << "| "
                << std::setw(20) << curenergies.stvk << "| "
                << std::setw(20) << curenergies.stvk_s1_dir_sin << "| "
                << std::setw(20) << curenergies.stvk_s1_dir_tan << "| "
                << std::setw(20) << curenergies.stvk_s2_dir_sin << "| "
                << std::setw(20) << curenergies.stvk_s2_dir_tan << std::endl;
            triangleArea *= multiplier;
            makeHalfCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);
            mesh = LibShell::MeshConnectivity(F);
        }
    }
    
}
