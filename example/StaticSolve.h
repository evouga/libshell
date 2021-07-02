#ifndef STATICSOLVE_H
#define STATICSOLVE_H

#include <Eigen/Core>
#include <vector>

#include "libshell/MaterialModel.h"
#include "libshell/MeshConnectivity.h"
#include "libshell/ElasticShell.h"

template <class SFF>
void takeOneStep(const LibShell::MeshConnectivity &mesh,
    Eigen::MatrixXd &curPos,
    Eigen::VectorXd &curEdgeDOFs,
    const LibShell::MaterialModel<SFF> &mat,
    const LibShell::RestState &restState,
    double &reg)
{

    int nverts = (int)curPos.rows();
    int nedges = mesh.nEdges();
    int nedgedofs = SFF::numExtraDOFs;

    int freeDOFs = 3 * nverts + nedgedofs * nedges;

    while (true)
    {
        Eigen::VectorXd derivative;
        std::vector<Eigen::Triplet<double> > hessian;

        double energy = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, curPos, curEdgeDOFs, mat, restState, &derivative, &hessian);

        Eigen::SparseMatrix<double> H(freeDOFs, freeDOFs);
        H.setFromTriplets(hessian.begin(), hessian.end());

        Eigen::VectorXd force = -derivative;
        Eigen::SparseMatrix<double> I(freeDOFs, freeDOFs);
        I.setIdentity();
        H += reg * I;
        
        Eigen::VectorXd maxvals(freeDOFs);
        maxvals.setZero();
        for (int k = 0; k < H.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it)
            {
                maxvals[it.row()] = std::max(maxvals[it.row()], std::fabs(it.value()));
            }
        }
        std::vector<Eigen::Triplet<double> > Dcoeffs;
        for (int i = 0; i < freeDOFs; i++)
        {
            double val = (maxvals[i] == 0.0 ? 1.0 : 1.0 / std::sqrt(maxvals[i]));            
            Dcoeffs.push_back({ i,i, val });
        }
        Eigen::SparseMatrix<double> D(freeDOFs, freeDOFs);
        D.setFromTriplets(Dcoeffs.begin(), Dcoeffs.end());

        Eigen::SparseMatrix<double> DHDT = D * H * D.transpose();

        std::cout << "solving, original force residual: " << force.norm() << std::endl;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(DHDT);
        if (solver.info() == Eigen::Success)
        {
            Eigen::VectorXd rhs = D * force;
            Eigen::VectorXd descentDir = D * solver.solve(rhs);
            std::cout << "solved" << std::endl;
            Eigen::MatrixXd newPos = curPos;
            for (int i = 0; i < nverts; i++)
            {
                newPos.row(i) += descentDir.segment<3>(3 * i);
            }
            Eigen::VectorXd newEdgeDofs = curEdgeDOFs + descentDir.segment(3 * nverts, nedgedofs * nedges);



            double newenergy = LibShell::ElasticShell<SFF>::elasticEnergy(mesh, newPos, newEdgeDofs, mat, restState, &derivative, NULL);
            force = -derivative;

            double forceResidual = force.norm();

            if (newenergy <= energy)
            {
                std::cout << "Old energy: " << energy << " new energy: " << newenergy << " force residual " << forceResidual << " pos change " << descentDir.segment(0, 3 * nverts).norm() << " theta change " << descentDir.segment(3 * nverts, nedgedofs*nedges).norm() << " lambda " << reg << std::endl;
                curPos = newPos;
                curEdgeDOFs = newEdgeDofs;
                reg /= 2.0;
                break;
            }
            else
            {
                std::cout << "Not a descent direction; old energy: " << energy << " new energy: " << newenergy << " lambda now: " << 2.0*reg << std::endl;
            }
        }
        else
        {
            std::cout << "Matrix not positive-definite, lambda now " << reg << std::endl;
        }

        reg *= 2.0;

    }
}

#endif