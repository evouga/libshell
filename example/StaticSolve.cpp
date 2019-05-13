#include "StaticSolve.h"
#include "../include/ElasticShell.h"
#include <iostream>
#include <Eigen/SparseCore>
#include "../include/MeshConnectivity.h"
#include "../include/SecondFundamentalFormDiscretization.h"

void takeOneStep(const MeshConnectivity &mesh, 
    Eigen::MatrixXd &curPos, 
    Eigen::VectorXd &curEdgeDOFs, 
    const MaterialModel &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    const std::vector<Eigen::Matrix2d> &bbars,
    const SecondFundamentalFormDiscretization &sff, 
    double &reg)
{
    
    int nverts = (int)curPos.rows();
    int nedges = mesh.nEdges();
    int nedgedofs = sff.numExtraDOFs();

    int freeDOFs = 3 * nverts + nedgedofs * nedges;
    
    while (true)
    {
        Eigen::VectorXd derivative;
        std::vector<Eigen::Triplet<double> > hessian;
                
        double energy = elasticEnergy(mesh, curPos, curEdgeDOFs, mat, thicknesses, abars, bbars, sff, &derivative, &hessian);
        
        Eigen::SparseMatrix<double> H(3 * nverts + nedgedofs * nedges, 3 * nverts + nedgedofs * nedges);
        H.setFromTriplets(hessian.begin(), hessian.end());

        Eigen::VectorXd force = -derivative;
        Eigen::SparseMatrix<double> I(freeDOFs, freeDOFs);
        I.setIdentity();
        H += reg * I;                
        std::cout << "solving" << std::endl;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(H);
        if (solver.info() == Eigen::Success)
        {
            Eigen::VectorXd descentDir = solver.solve(force);
            std::cout << "solved" << std::endl;
            Eigen::MatrixXd newPos = curPos;
            for (int i = 0; i < nverts; i++)
            {
                newPos.row(i) += descentDir.segment<3>(3 * i);
            }
            Eigen::VectorXd newEdgeDofs = curEdgeDOFs + descentDir.segment(3 * nverts, nedgedofs * nedges);



            double newenergy = elasticEnergy(mesh, newPos, newEdgeDofs, mat, thicknesses, abars, bbars, sff, &derivative, NULL);
            force = -derivative;

            double forceResidual = force.norm();

            if (newenergy <= energy)
            {
                std::cout << "Old energy: " << energy << " new energy: " << newenergy << " force residual " << forceResidual << " pos change " << descentDir.segment(0, 3 * nverts).norm() << " theta change " << descentDir.segment(3 * nverts, nedgedofs*nedges).norm() << std::endl;
                curPos = newPos;
                curEdgeDOFs = newEdgeDofs;
                reg /= 2.0;
                break;
            }
            else
            {
                std::cout << "Not a descent direction; old energy: " << energy << " new energy: " << newenergy << " lambda now: " << reg << std::endl;        
            }
        }
        else
        {
            std::cout << "Matrix not positive-definite" << std::endl;
        }

        reg *= 2.0;
        
    }           
}
