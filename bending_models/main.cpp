#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "make_geometric_shapes/HalfCylinder.h"
#include "make_geometric_shapes/Sphere.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "../include/StVKMaterial.h"
#include "igl/readOBJ.h"
#include <set>
#include <vector>
#include "ShellEnergy.h"
#include "igl/writePLY.h"
#include "polyscope/surface_vector_quantity.h"
#include "igl/principal_curvature.h"

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
    double exact;
    double quadraticbending;
    double neohookean;
    double neohookeandir;
    double neohookeandirzero;
    double stvk;
    double stvkdir;
    double prism;
};

void optimizeEdgeDOFs(ShellEnergy& energy, const Eigen::MatrixXd& curPos, Eigen::VectorXd& edgeDOFs)
{
    double tol = 1e-5;
    int nposdofs = curPos.rows() * 3;
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

    double reg = 1e-6;
    while (true)
    {
        std::vector<Eigen::Triplet<double> > Hcoeffs;
        Eigen::VectorXd F;
        double origEnergy = energy.elasticEnergy(curPos, edgeDOFs, true, &F, &Hcoeffs);
        Eigen::VectorXd PF = P * F;
        std::cout << "Force resid now: " << PF.norm() << ", energy: " << origEnergy << ", reg: " << reg << std::endl;
        if (PF.norm() < tol)
            return;
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
        double newenergy = energy.elasticEnergy(curPos, newedgeDOFs, true, NULL, NULL);
        if (newenergy > origEnergy)
        {
            std::cout << "Not a descent step, " << origEnergy << " -> " << newenergy << std::endl;
            reg *= 2.0;
            continue;
        }
        edgeDOFs = newedgeDOFs;
        reg *= 0.5;
    }
}


Energies measureCylinderEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    const Eigen::MatrixXd& curPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    double curRadius,
    double curHeight,
    Eigen::MatrixXd &nhForces,
    Eigen::MatrixXd &qbForces) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh,
                                                             restPos);

    Eigen::VectorXd zerodiredgeDOFs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zerodiredgeDOFs, mesh,
                                                              restPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;
    LibShell::MonolayerRestState dirrestState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);
    dirrestState.thicknesses.resize(mesh.nFaces(), thickness);
    dirrestState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    dirrestState.lameBeta.resize(mesh.nFaces(), lameBeta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        firstFundamentalForms(mesh, restPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        secondFundamentalForms(mesh, restPos, edgeDOFs, restState.bbars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        firstFundamentalForms(mesh, restPos, dirrestState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        secondFundamentalForms(mesh, restPos, zerodiredgeDOFs, dirrestState.bbars);

    std::vector<Eigen::Matrix2d> cur_bs;
    LibShell::ElasticShell<
        LibShell::MidedgeAverageFormulation>::secondFundamentalForms(mesh,
                                                                     curPos,
                                                                     edgeDOFs,
                                                                     cur_bs);

    // Make the half-cylinder rest-flat
    for (int i = 0; i < mesh.nFaces(); i++)
        restState.bbars[i].setZero();

    Eigen::VectorXd restEdgeDOFs = edgeDOFs;

    NeohookeanShellEnergy nhenergyModel(mesh, restState);
    NeohookeanDirectorShellEnergy nhdenergyModel(mesh, dirrestState);
    QuadraticBendingShellEnergy qbenergyModel(mesh, restState, restPos,
                                              restEdgeDOFs);
    StVKShellEnergy stvkenergyModel(mesh, restState);
    StVKDirectorShellEnergy stvkdenergyModel(mesh, dirrestState);

    Eigen::VectorXd nhF, stvkF;
    Eigen::VectorXd nhdF, stvkdF;
    Eigen::VectorXd qbF;

    Eigen::VectorXd diredgeDOFs = zerodiredgeDOFs;
    optimizeEdgeDOFs(nhdenergyModel, curPos, diredgeDOFs);

    result.neohookean =
        nhenergyModel.elasticEnergy(curPos, edgeDOFs, true, &nhF, NULL);
    result.neohookeandir =
        nhdenergyModel.elasticEnergy(curPos, diredgeDOFs, true, &nhdF, NULL);
    result.neohookeandirzero =
        nhdenergyModel.elasticEnergy(curPos, zerodiredgeDOFs, true, &nhdF, NULL);
    result.quadraticbending =
        qbenergyModel.elasticEnergy(curPos, edgeDOFs, true, &qbF, NULL);
    result.stvk =
        stvkenergyModel.elasticEnergy(curPos, edgeDOFs, true, &stvkF, NULL);
    result.stvkdir = stvkdenergyModel.elasticEnergy(curPos, diredgeDOFs, true,
                                                    &stvkdF, NULL);

    int nverts = curPos.rows();
    nhForces.resize(nverts, 3);
    qbForces.resize(nverts, 3);
    for (int i = 0; i < nverts; i++) {
        nhForces.row(i) = -nhF.segment<3>(3 * i);
        qbForces.row(i) = -qbF.segment<3>(3 * i);
    }

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
    b << 1.0 / curRadius, 0, 0, 0;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm =
        lameAlpha / 2.0 * M.trace() * M.trace() + lameBeta * (M * M).trace();
    double coeff = thickness * thickness * thickness / 12.0;
    constexpr double PI = 3.1415926535898;
    double area = PI * curRadius * curHeight;

    result.exact = svnorm * coeff * area;

    return result;
}

Energies measureSphereEnergy(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    double radius) {
    Energies result;

    // initialize default edge DOFs (edge director angles)
    Eigen::VectorXd edgeDOFs;
    LibShell::MidedgeAverageFormulation::initializeExtraDOFs(edgeDOFs, mesh,
        curPos);

    Eigen::VectorXd zerodiredgeDOFs;
    LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(zerodiredgeDOFs, mesh,
        curPos);

    // initialize the rest geometry of the shell
    LibShell::MonolayerRestState restState;
    LibShell::MonolayerRestState dirrestState;

    // set uniform thicknesses
    restState.thicknesses.resize(mesh.nFaces(), thickness);
    restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    restState.lameBeta.resize(mesh.nFaces(), lameBeta);
    dirrestState.thicknesses.resize(mesh.nFaces(), thickness);
    dirrestState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
    dirrestState.lameBeta.resize(mesh.nFaces(), lameBeta);

    // initialize first and second fundamental forms to those of input mesh
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        firstFundamentalForms(mesh, curPos, restState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::
        secondFundamentalForms(mesh, curPos, edgeDOFs, restState.bbars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        firstFundamentalForms(mesh, curPos, dirrestState.abars);
    LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
        secondFundamentalForms(mesh, curPos, zerodiredgeDOFs, dirrestState.bbars);


    // Make the half-cylinder rest-flat
    for (int i = 0; i < mesh.nFaces(); i++)
    {
        restState.bbars[i].setZero();
        dirrestState.bbars[i].setZero();
    }

    Eigen::VectorXd restEdgeDOFs = edgeDOFs;

    NeohookeanShellEnergy nhenergyModel(mesh, restState);
    NeohookeanDirectorShellEnergy nhdenergyModel(mesh, dirrestState);
    QuadraticBendingShellEnergy qbenergyModel(mesh, restState, curPos,
        restEdgeDOFs);
    StVKShellEnergy stvkenergyModel(mesh, restState);
    StVKDirectorShellEnergy stvkdenergyModel(mesh, dirrestState);

    Eigen::VectorXd diredgeDOFs = zerodiredgeDOFs;
    optimizeEdgeDOFs(nhdenergyModel, curPos, diredgeDOFs);

    result.neohookean =
        nhenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.neohookeandir =
        nhdenergyModel.elasticEnergy(curPos, diredgeDOFs, true, NULL, NULL);
    result.neohookeandirzero =
        nhdenergyModel.elasticEnergy(curPos, zerodiredgeDOFs, true, NULL, NULL);
    result.quadraticbending =
        qbenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvk =
        stvkenergyModel.elasticEnergy(curPos, edgeDOFs, true, NULL, NULL);
    result.stvkdir = stvkdenergyModel.elasticEnergy(curPos, diredgeDOFs, true,
        NULL, NULL);

    // ground truth energy
    Eigen::Matrix2d abar;
    abar.setIdentity();

    Eigen::Matrix2d b;
    b << 1.0 / radius, 0, 0, 1.0 / radius;

    Eigen::Matrix2d M = abar.inverse() * b;
    double svnorm =
        lameAlpha / 2.0 * M.trace() * M.trace() + lameBeta * (M * M).trace();
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

    triangleArea = 0.0000001;

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

    double lameAlpha, lameBeta;
    lameParameters(lameAlpha, lameBeta);
    double curRadius = cokeRadius;
    double curHeight = cokeHeight;

    int steps = 5;
    double multiplier = 4;
    std::ofstream log("log.txt");
    log << "#V, exact energy, NH energy, NH_dir energy, NH_dir_zero energy, StVK energy, StVK_dir energy, quadratic, prism" << std::endl;
    if (curMeshType == MeshType::MT_SPHERE)
    {
        makeSphere(sphereRadius, triangleArea, origV, F);
        LibShell::MeshConnectivity mesh(F);
        for (int step = 0; step < steps; step++)
        {
            std::stringstream ss;
            ss << "sphere_ " << step << ".ply";
            igl::writePLY(ss.str(), origV, F);

            curenergies = measureSphereEnergy(mesh, origV, thickness, lameAlpha, lameBeta, sphereRadius);
            log << origV.rows() << ": " << curenergies.exact << " " << curenergies.neohookean << " " << curenergies.neohookeandir << " " << curenergies.neohookeandirzero << " " << curenergies.stvk << " " << curenergies.stvkdir << " " << curenergies.quadraticbending << " " << curenergies.prism << std::endl;
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
            curenergies = measureCylinderEnergy(mesh, origV, rolledV, thickness, lameAlpha, lameBeta, curRadius, curHeight, nhForces, qbForces);
            log << origV.rows() << ": " << curenergies.exact << " "
                << curenergies.neohookean << " " << curenergies.neohookeandir
                << " " << curenergies.neohookeandir << " " << curenergies.stvk
                << " " << curenergies.stvkdir << " "
                << curenergies.quadraticbending << " " << curenergies.prism
                << std::endl;
            triangleArea *= multiplier;
            makeHalfCylinder(curMeshType == MeshType::MT_CYLINDER_REGULAR, cokeRadius, cokeHeight, triangleArea, origV, rolledV, F);
            mesh = LibShell::MeshConnectivity(F);
        }
    }
    
}
