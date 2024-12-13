#ifndef SHELLENERGY_H
#define SHELLENERGY_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include "../include/MeshConnectivity.h"
#include "../include/RestState.h"
#include "../include/MaterialModel.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/StVKMaterial.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/MidedgeAngleCompressiveFormulation.h"
#include "../include/MidedgeAngleGeneralSinFormulation.h"
#include "../include/MidedgeAngleGeneralTanFormulation.h"
#include "../include/MidedgeAngleGeneralFormulation.h"
#include "../include/ElasticShell.h"
#include "QuadraticExpansionBending.h"
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

static Eigen::Matrix3d crossMatrix(Eigen::Vector3d v) {
    Eigen::Matrix3d ret;
    ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return ret;
}

static double angle(const Eigen::Vector3d& v,
                    const Eigen::Vector3d& w,
                    const Eigen::Vector3d& axis,
                    Eigen::Matrix<double, 1, 9>* derivative,  // v, w
                    Eigen::Matrix<double, 9, 9>* hessian) {
    double theta = 2.0 * atan2((v.cross(w).dot(axis) / axis.norm()), v.dot(w) + v.norm() * w.norm());

    if (derivative) {
        derivative->segment<3>(0) = -axis.cross(v) / v.squaredNorm() / axis.norm();
        derivative->segment<3>(3) = axis.cross(w) / w.squaredNorm() / axis.norm();
        derivative->segment<3>(6).setZero();
    }
    if (hessian) {
        hessian->setZero();
        hessian->block<3, 3>(0, 0) +=
            2.0 * (axis.cross(v)) * v.transpose() / v.squaredNorm() / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) +=
            -2.0 * (axis.cross(w)) * w.transpose() / w.squaredNorm() / w.squaredNorm() / axis.norm();
        hessian->block<3, 3>(0, 0) += -crossMatrix(axis) / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) += crossMatrix(axis) / w.squaredNorm() / axis.norm();

        Eigen::Matrix3d dahat = (Eigen::Matrix3d::Identity() / axis.norm() -
                                 axis * axis.transpose() / axis.norm() / axis.norm() / axis.norm());

        hessian->block<3, 3>(0, 6) += crossMatrix(v) * dahat / v.squaredNorm();
        hessian->block<3, 3>(3, 6) += -crossMatrix(w) * dahat / w.squaredNorm();
    }

    return theta;
}

class ShellEnergy {
public:
    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const = 0;
};

class NeohookeanShellEnergy : public ShellEnergy {
public:
    NeohookeanShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAverageFormulation> mat_;
};

class NeohookeanS1DirectorSinShellEnergy : public ShellEnergy {
public:
    NeohookeanS1DirectorSinShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleSinFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleSinFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleSinFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAngleSinFormulation> mat_;
};

class NeohookeanS1DirectorTanShellEnergy : public ShellEnergy {
public:
    NeohookeanS1DirectorTanShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAngleTanFormulation> mat_;
};

class NeohookeanS2DirectorSinShellEnergy : public ShellEnergy {
public:
    NeohookeanS2DirectorSinShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleGeneralSinFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleGeneralSinFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleGeneralSinFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAngleGeneralSinFormulation> mat_;
};

class NeohookeanS2DirectorTanShellEnergy : public ShellEnergy {
public:
    NeohookeanS2DirectorTanShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAngleGeneralTanFormulation> mat_;
};

class NeohookeanCompressiveDirectorShellEnergy : public ShellEnergy {
public:
    NeohookeanCompressiveDirectorShellEnergy(const LibShell::MeshConnectivity& mesh,
                                             const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleCompressiveFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |=
                LibShell::ElasticShell<LibShell::MidedgeAngleCompressiveFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleCompressiveFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAngleCompressiveFormulation> mat_;
};

class StVKShellEnergy : public ShellEnergy {
public:
    StVKShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAverageFormulation> mat_;
};

class StVKS1DirectorSinShellEnergy : public ShellEnergy {
public:
    StVKS1DirectorSinShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleSinFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleSinFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleSinFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleSinFormulation> mat_;
};

class StVKS1DirectorTanShellEnergy : public ShellEnergy {
public:
    StVKS1DirectorTanShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleTanFormulation> mat_;
};

class StVKS2DirectorSinShellEnergy : public ShellEnergy {
public:
    StVKS2DirectorSinShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleGeneralSinFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleGeneralSinFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleGeneralSinFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleGeneralSinFormulation> mat_;
};

class StVKS2DirectorTanShellEnergy : public ShellEnergy {
public:
    StVKS2DirectorTanShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleGeneralTanFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleGeneralTanFormulation> mat_;
};

class StVKGeneralDirectorShellEnergy : public ShellEnergy {
public:
    StVKGeneralDirectorShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |= LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleGeneralFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleGeneralFormulation> mat_;
};

class StVKCompressiveDirectorShellEnergy : public ShellEnergy {
public:
    StVKCompressiveDirectorShellEnergy(const LibShell::MeshConnectivity& mesh, const LibShell::RestState& restState)
        : mesh_(mesh),
          restState_(restState),
          mat_() {}

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        int whichTerms = LibShell::ElasticShell<LibShell::MidedgeAngleCompressiveFormulation>::EnergyTerm::ET_BENDING;
        if (!bendingOnly)
            whichTerms |=
                LibShell::ElasticShell<LibShell::MidedgeAngleCompressiveFormulation>::EnergyTerm::ET_STRETCHING;
        return LibShell::ElasticShell<LibShell::MidedgeAngleCompressiveFormulation>::elasticEnergy(
            mesh_, curPos, curEdgeDOFs, mat_, restState_, whichTerms, derivative, hessian, proj_type);
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    LibShell::StVKMaterial<LibShell::MidedgeAngleCompressiveFormulation> mat_;
};

class QuadraticExpansionShellEnergy : public ShellEnergy {
public:
    QuadraticExpansionShellEnergy(const LibShell::MeshConnectivity& mesh,
                                  const LibShell::RestState& restState,
                                  const Eigen::MatrixXd& restPos,
                                  const Eigen::VectorXd& restEdgeDOFs)
        : mesh_(mesh),
          restState_(restState),
          mat_(),
          restPos_(restPos),
          restEdgeDOFs_(restEdgeDOFs) {
        int nverts = restPos.rows();
        bendingMatrix<LibShell::MidedgeAverageFormulation>(mesh, restPos, restEdgeDOFs, restState, bendingMcoeffs_);
        bendingM_.resize(3 * nverts, 3 * nverts);
        bendingM_.setFromTriplets(bendingMcoeffs_.begin(), bendingMcoeffs_.end());
    }

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        double result = 0;
        int nverts = curPos.rows();

        if (derivative) {
            derivative->resize(3 * nverts);
            derivative->setZero();
        }
        if (!bendingOnly)
            result += LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(
                mesh_, curPos, curEdgeDOFs, mat_, restState_,
                LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING, derivative,
                hessian, proj_type);
        Eigen::VectorXd displacement(3 * nverts);
        for (int i = 0; i < nverts; i++) {
            for (int j = 0; j < 3; j++) {
                displacement[3 * i + j] = curPos(i, j) - restPos_(i, j);
            }
        }

        double bendingEnergy = 0.5 * displacement.transpose() * bendingM_ * displacement;
        result += bendingEnergy;

        if (derivative) {
            *derivative += bendingM_ * displacement;
        }
        if (hessian) {
            for (auto it : bendingMcoeffs_) hessian->push_back(it);
        }

        return result;
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    const Eigen::MatrixXd& restPos_;
    const Eigen::VectorXd& restEdgeDOFs_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAverageFormulation> mat_;
    std::vector<Eigen::Triplet<double>> bendingMcoeffs_;
    Eigen::SparseMatrix<double> bendingM_;
};

class QuadraticBendingShellEnergy : public ShellEnergy {
public:
    QuadraticBendingShellEnergy(const LibShell::MeshConnectivity& mesh,
                                const LibShell::RestState& restState,
                                const Eigen::MatrixXd& restPos,
                                const Eigen::VectorXd& restEdgeDOFs,
                                LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone)
        : mesh_(mesh),
          restState_(restState),
          mat_(),
          restPos_(restPos),
          restEdgeDOFs_(restEdgeDOFs) {
        int nverts = restPos.rows();
        int nfaces = mesh.nFaces();
        int nedges = mesh.nEdges();

        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(restPos, mesh.faces(), L);
        std::vector<Eigen::Triplet<double>> bigLcoeffs;
        for (int k = 0; k < L.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                for (int j = 0; j < 3; j++) {
                    bigLcoeffs.push_back({3 * (int)it.row() + j, 3 * (int)it.col() + j, it.value()});
                }
            }
        }

        Eigen::SparseMatrix<double> bigL(3 * nverts, 3 * nverts);
        bigL.setFromTriplets(bigLcoeffs.begin(), bigLcoeffs.end());

        std::vector<bool> bdry(nverts);
        std::vector<Eigen::Triplet<double>> Ncoeffs;
        for (int i = 0; i < nedges; i++) {
            for (int side = 0; side < 2; side++) {
                if (mesh.edgeFace(i, 1 - side) == -1) {
                    int oppvidx = mesh.edgeOppositeVertex(i, side);
                    int v0idx = mesh.edgeVertex(i, 0);
                    int v1idx = mesh.edgeVertex(i, 1);
                    bdry[v0idx] = true;
                    bdry[v1idx] = true;
                    Eigen::Vector3d oppv = restPos.row(oppvidx).transpose();
                    Eigen::Vector3d v0 = restPos.row(v0idx).transpose();
                    Eigen::Vector3d v1 = restPos.row(v1idx).transpose();
                    double cot0 = (oppv - v0).dot(v1 - v0) / (oppv - v0).cross(v1 - v0).norm();
                    double cot1 = (oppv - v1).dot(v0 - v1) / (oppv - v1).cross(v0 - v1).norm();
                    for (int j = 0; j < 3; j++) {
                        Ncoeffs.push_back({3 * v0idx + j, 3 * v0idx + j, 0.5 * cot1});
                        Ncoeffs.push_back({3 * v0idx + j, 3 * v1idx + j, 0.5 * cot0});
                        Ncoeffs.push_back({3 * v0idx + j, 3 * oppvidx + j, 0.5 * (-cot0 - cot1)});
                        Ncoeffs.push_back({3 * v1idx + j, 3 * v0idx + j, 0.5 * cot1});
                        Ncoeffs.push_back({3 * v1idx + j, 3 * v1idx + j, 0.5 * cot0});
                        Ncoeffs.push_back({3 * v1idx + j, 3 * oppvidx + j, 0.5 * (-cot0 - cot1)});
                    }
                }
            }
        }
        Eigen::SparseMatrix<double> N(3 * nverts, 3 * nverts);
        N.setFromTriplets(Ncoeffs.begin(), Ncoeffs.end());

        std::vector<double> Mcoeffs(nverts);
        std::vector<double> energycoeffs(nverts);
        std::vector<Eigen::Triplet<double>> Minvcoeffs;

        for (int i = 0; i < nfaces; i++) {
            double h = ((LibShell::MonolayerRestState&)restState).thicknesses[i];
            double lameAlpha = ((LibShell::MonolayerRestState&)restState).lameAlpha[i];
            double lameBeta = ((LibShell::MonolayerRestState&)restState).lameBeta[i];
            double weight = h * h * h / 12.0 * (lameAlpha + 2.0 * lameBeta);
            double area = 0.5 * std::sqrt(((LibShell::MonolayerRestState&)restState).abars[i].determinant());

            for (int j = 0; j < 3; j++) {
                int vidx = mesh.faceVertex(i, j);
                Mcoeffs[vidx] += area / 3.0;
                energycoeffs[vidx] += weight * area / 3.0;
            }
        }
        for (int i = 0; i < nverts; i++) {
            for (int j = 0; j < 3; j++) {
                Minvcoeffs.push_back({3 * i + j, 3 * i + j, energycoeffs[i] / Mcoeffs[i] / Mcoeffs[i]});
            }
        }
        Eigen::SparseMatrix<double> Minv(3 * nverts, 3 * nverts);
        Minv.setFromTriplets(Minvcoeffs.begin(), Minvcoeffs.end());

        Eigen::SparseMatrix<double> biL = (bigL.transpose() + N.transpose()) * Minv * (bigL + N);

        bendingMcoeffs_.clear();
        for (int k = 0; k < biL.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(biL, k); it; ++it) {
                bendingMcoeffs_.push_back({(int)it.row(), (int)it.col(), it.value()});
            }
        }

        bendingM_.resize(3 * nverts, 3 * nverts);
        bendingM_.setFromTriplets(bendingMcoeffs_.begin(), bendingMcoeffs_.end());
    }

    virtual double elasticEnergy(const Eigen::MatrixXd& curPos,
                                 const Eigen::VectorXd& curEdgeDOFs,
                                 bool bendingOnly,
                                 Eigen::VectorXd* derivative,  // positions, then thetas
                                 std::vector<Eigen::Triplet<double>>* hessian,
                                 LibShell::HessianProjectType proj_type = LibShell::HessianProjectType::kNone) const {
        double result = 0;
        int nverts = curPos.rows();

        if (derivative) {
            derivative->resize(3 * nverts);
            derivative->setZero();
        }

        if (!bendingOnly)
            result += LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::elasticEnergy(
                mesh_, curPos, curEdgeDOFs, mat_, restState_,
                LibShell::ElasticShell<LibShell::MidedgeAverageFormulation>::EnergyTerm::ET_STRETCHING, derivative,
                hessian, proj_type);

        Eigen::VectorXd displacement(3 * nverts);
        for (int i = 0; i < nverts; i++) {
            for (int j = 0; j < 3; j++) {
                displacement[3 * i + j] = curPos(i, j);
            }
        }

        double bendingEnergy = 0.5 * displacement.transpose() * bendingM_ * displacement;
        result += bendingEnergy;

        if (derivative) {
            *derivative += bendingM_ * displacement;
        }
        if (hessian) {
            for (auto it : bendingMcoeffs_) hessian->push_back(it);
        }

        return result;
    }

    const LibShell::MeshConnectivity& mesh_;
    const LibShell::RestState& restState_;
    const Eigen::MatrixXd& restPos_;
    const Eigen::VectorXd& restEdgeDOFs_;
    LibShell::NeoHookeanMaterial<LibShell::MidedgeAverageFormulation> mat_;
    std::vector<Eigen::Triplet<double>> bendingMcoeffs_;
    Eigen::SparseMatrix<double> bendingM_;
};


#endif