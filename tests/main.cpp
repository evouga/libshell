#include <Eigen/Core>
#include <iostream>
#include <map>
#include <cmath>
#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "findiff.h"
#include <random>

std::default_random_engine rng;

const int nummats = 2;
const int numsff = 3;    


template<class SFF>
static void testStretchingFiniteDifferences(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    bool verbose,
    FiniteDifferenceLog &log);

template<class SFF>
static void testBendingFiniteDifferences(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &edgeDOFs,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    const std::vector<Eigen::Matrix2d> &bbars,
    bool verbose,
    FiniteDifferenceLog &log);


void makeSquareMesh(int dim, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    V.resize(dim*dim, 3);
    F.resize(2 * (dim - 1) * (dim - 1), 3);
    int vrow = 0;
    int frow = 0;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            double y = -(i * 1.0 + (dim - i - 1) * -1.0) / double(dim - 1);
            double x = (j * 1.0 + (dim - j - 1) * -1.0) / double(dim - 1);

            V(vrow, 0) = x;
            V(vrow, 1) = y;
            V(vrow, 2) = 0;
            vrow++;

            if (i != 0 && j != 0)
            {
                int iprev = i - 1;
                int jprev = j - 1;
                F(frow, 0) = iprev * dim + jprev;
                F(frow, 1) = iprev * dim + j;
                F(frow, 2) = i * dim + j;
                frow++;
                F(frow, 0) = iprev * dim + jprev;
                F(frow, 1) = i * dim + j;
                F(frow, 2) = i * dim + jprev;
                frow++;
            }
        }
    }
}

void printDiffLog(const std::map<int, double> &difflog)
{
    for (auto it : difflog)
    {
        std::cout << it.first << "\t" << it.second << std::endl;
    }
}

template <class SFF>
void differenceTest(const MeshConnectivity &mesh,
    const Eigen::MatrixXd &restPos,
    int matid,
    bool verbose)
{
    Eigen::MatrixXd curPos = restPos;
    curPos.setRandom();
    const double PI = 3.14159263535898;

    Eigen::VectorXd edgeDOFs;
    SFF::initializeExtraDOFs(edgeDOFs, mesh, curPos);
    int nedgeDOFs = (int)edgeDOFs.size();
    std::uniform_real_distribution<double> angdist(-PI / 2, PI / 2);
    for (int i = 0; i < nedgeDOFs; i++)
    {
        edgeDOFs[i] = angdist(rng);
    }
    std::vector<Eigen::Matrix2d> abar;
    ElasticShell<SFF>::firstFundamentalForms(mesh, curPos, abar);

    std::vector<Eigen::Matrix2d> bbar;
    ElasticShell<SFF>::secondFundamentalForms(mesh, curPos, edgeDOFs, bbar);

    std::uniform_real_distribution<double> logThicknessDist(-6, 0);
    int nfaces = mesh.nFaces();
    Eigen::VectorXd thicknesses(nfaces);
    for (int i = 0; i < nfaces; i++)
    {
        thicknesses[i] = std::pow(10.0, logThicknessDist(rng));
    }

    std::uniform_real_distribution<double> loglamedist(-1, 1);

    for (int lameiter = 0; lameiter < 2; lameiter++)
    {
        double lameAlpha = 0;
        double lameBeta = 0;
        (lameiter == 1 ? lameAlpha : lameBeta) = std::pow(10.0, loglamedist(rng));

        MaterialModel<SFF> *mat;
        switch (matid)
        {
        case 0:
            std::cout << "NeoHookeanMaterial, ";
            mat = new NeoHookeanMaterial<SFF>(lameAlpha, lameBeta);
            break;
        case 1:
            std::cout << "StVKMaterial, ";
            mat = new StVKMaterial<SFF>(lameAlpha, lameBeta);
            break;
        default:
            assert(false);
        }

        std::cout << "alpha = " << lameAlpha << ", beta = " << lameBeta << std::endl;

        curPos.setRandom();
        for (int i = 0; i < nedgeDOFs; i++)
        {
            edgeDOFs[i] = angdist(rng);
        }

        FiniteDifferenceLog log;
        testStretchingFiniteDifferences(mesh, curPos, *mat, thicknesses, abar, verbose, log);
        std::cout << "Stretching:" << std::endl;
        log.printStats();
        testBendingFiniteDifferences(mesh, curPos, edgeDOFs, *mat, thicknesses, abar, bbar, verbose, log);
        std::cout << "Bending:" << std::endl;
        log.printStats();
        std::cout << std::endl;

        delete mat;
    }
}


template<class SFF> 
void getHessian(const MeshConnectivity &mesh, 
    const Eigen::MatrixXd &curPos, 
    const Eigen::VectorXd &thicknesses, 
    int matid,
    double lameAlpha, double lameBeta,
    Eigen::SparseMatrix<double> &H)
{
    Eigen::VectorXd edgeDOFs;
    SFF::initializeExtraDOFs(edgeDOFs, mesh, curPos);
    int nedgeDOFs = (int)edgeDOFs.size();
    std::vector<Eigen::Matrix2d> abar;
    ElasticShell<SFF>::firstFundamentalForms(mesh, curPos, abar);

    std::vector<Eigen::Matrix2d> bbar;
    ElasticShell<SFF>::secondFundamentalForms(mesh, curPos, edgeDOFs, bbar);

    std::vector<Eigen::Triplet<double> > hessian;

    MaterialModel<SFF> *mat;
    switch (matid)
    {
    case 0:
        mat = new NeoHookeanMaterial<SFF>(lameAlpha, lameBeta);
        break;
    case 1:
        mat = new StVKMaterial<SFF>(lameAlpha, lameBeta);
        break;
    default:
        assert(false);
    }

    ElasticShell<SFF>::elasticEnergy(mesh, curPos, edgeDOFs, *mat, thicknesses, abar, bbar, NULL, &hessian);

    int nverts = curPos.rows();
    int nedges = mesh.nEdges();
    int dim = 3 * nverts + nedgeDOFs;
    H.resize(dim, dim);
    H.setFromTriplets(hessian.begin(), hessian.end());

    delete mat;
}


void consistencyTests(const MeshConnectivity &mesh, const Eigen::MatrixXd &restPos)
{
    std::uniform_real_distribution<double> logThicknessDist(-6, 0);
    int nfaces = mesh.nFaces();
    Eigen::VectorXd thicknesses(nfaces);
    for (int i = 0; i < nfaces; i++)
    {
        thicknesses[i] = std::pow(10.0, logThicknessDist(rng));
    }

    std::uniform_real_distribution<double> loglamedist(-1, 1);

    for (int lameiter = 0; lameiter < 2; lameiter++)
    {
        double lameAlpha = 0;
        double lameBeta = 0;
        (lameiter == 1 ? lameAlpha : lameBeta) = std::pow(10.0, loglamedist(rng));
        std::cout << "Testing with alpha = " << lameAlpha << ", beta = " << lameBeta << std::endl;
        
        int numoptions = nummats * numsff;
        std::vector<Eigen::SparseMatrix<double> > hessians(numoptions);
        for (int i = 0; i < nummats; i++)
        {
            for (int j = 0; j < numsff; j++)
            {
                switch (j)
                {
                case 0:
                    getHessian<MidedgeAngleTanFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta, hessians[i*numsff + j]);
                    break;
                case 1:
                    getHessian<MidedgeAngleSinFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta, hessians[i*numsff + j]);
                    break;
                case 2:
                    getHessian<MidedgeAverageFormulation>(mesh, restPos, thicknesses, i, lameAlpha, lameBeta, hessians[i*numsff + j]);
                    break;
                default:
                    assert(false);
                }                
            }
        }
        for (int i = 0; i < nummats; i++)
        {
            for (int j = 0; j < numsff; j++)
            {
                for (int k = 0; k < nummats; k++)
                {
                    for (int l = 0; l < numsff; l++)
                    {
                        int idx1 = i * numsff + j;
                        int idx2 = k * numsff + l;
                        if (idx2 <= idx1)
                            continue;
                        std::string matnames[] = { "Neohk", "StVK" };
                        std::string sffnames[] = { "Tan", "Sin", "Avg" };
                        std::cout << "(" << matnames[i] << ", " << sffnames[j] << ") vs (" << matnames[k] << ", " << sffnames[l] << "): ";
                        double diff = 0;
                        int nverts = restPos.rows();
                        for (int m = 0; m < 3 * nverts; m++)
                        {
                            for (int n = m; n < 3 * nverts; n++)
                            {
                                diff += std::fabs(hessians[idx1].coeff(m, n) - hessians[idx2].coeff(m, n));
                            }
                        }
                        std::cout << diff << std::endl;
                    }                    
                }
            }
        }
    }
}


int main()
{
    int dim = 20;
    bool verbose = false;
    bool testderivatives = true;
    bool testconsistency = true;
    
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    makeSquareMesh(dim, V, F);

    MeshConnectivity mesh(F);
    std::default_random_engine generator;

    if (testderivatives)
    {
        std::cout << "Running finite difference tests" << std::endl;;
        for (int i = 0; i < nummats; i++)
        {
            for (int j = 0; j < numsff; j++)
            {
                std::cout << "Starting trial: ";
                switch (j)
                {
                case 0:
                    std::cout << "MidedgeAngleTanFormulation, ";
                    differenceTest<MidedgeAngleTanFormulation>(mesh, V, i, verbose);
                    break;
                case 1:
                    std::cout << "MidedgeAngleSinFormulation, ";
                    differenceTest<MidedgeAngleSinFormulation>(mesh, V, i, verbose);
                    break;
                case 2:
                    std::cout << "MidedgeAverageFormulation, ";
                    differenceTest<MidedgeAverageFormulation>(mesh, V, i, verbose);
                    break;
                default:
                    assert(false);
                }
            }
        }
        std::cout << "Finite difference tests done" << std::endl;
    }
    if (testconsistency)
    {
        std::cout << "Running consistency tests" << std::endl;;
        consistencyTests(mesh, V);
        std::cout << "Consistency tests done" << std::endl;
    }
}


template <class SFF>
void testStretchingFiniteDifferences(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    bool verbose,
    FiniteDifferenceLog &log)
{
    log.clear();
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    Eigen::MatrixXd testpos = curPos;
    testpos.setRandom();

    std::vector<int> epsilons = {-2, -3, -4, -5, -6};
    
    for (auto epsilon : epsilons)
    {
        double pert = std::pow(10.0, epsilon);

        for (int face = 0; face < nfaces; face++)
        {

            Eigen::Matrix<double, 1, 9> deriv;
            Eigen::Matrix<double, 9, 9> hess;
            double result = mat.stretchingEnergy(mesh, testpos, thicknesses[face], abars[face], face, &deriv, &hess);

            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Eigen::MatrixXd fwdpertpos = testpos;
                    Eigen::MatrixXd backpertpos = testpos;
                    fwdpertpos(mesh.faceVertex(face, j), k) += pert;
                    backpertpos(mesh.faceVertex(face, j), k) -= pert;

                    Eigen::Matrix<double, 1, 9> fwdpertderiv;
                    Eigen::Matrix<double, 1, 9> backpertderiv;
                    double fwdnewresult = mat.stretchingEnergy(mesh, fwdpertpos, thicknesses[face], abars[face], face, &fwdpertderiv, NULL);
                    double backnewresult = mat.stretchingEnergy(mesh, backpertpos, thicknesses[face], abars[face], face, &backpertderiv, NULL);
                    double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                    if(verbose) std::cout << '(' << j << ", " << k << ") " << findiff << " " << deriv(0, 3 * j + k) << std::endl;
                    log.addEntry(pert, deriv(0, 3 * j + k), findiff);
                    Eigen::Matrix<double, 1, 9> derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                    if (verbose)
                    {
                        std::cout << derivdiff << std::endl;
                        std::cout << "//" << std::endl;
                        std::cout << hess.row(3 * j + k) << std::endl << std::endl;
                    }
                    for (int l = 0; l < 9; l++)
                    {
                        log.addEntry(pert, hess(3 * j + k, l), derivdiff(0, l));
                    }                    
                }
            }
        }        
    }
}

template <class SFF>
void testBendingFiniteDifferences(
    const MeshConnectivity &mesh,
    const Eigen::MatrixXd &curPos,
    const Eigen::VectorXd &edgeDOFs,
    const MaterialModel<SFF> &mat,
    const Eigen::VectorXd &thicknesses,
    const std::vector<Eigen::Matrix2d> &abars,
    const std::vector<Eigen::Matrix2d> &bbars,
    bool verbose,
    FiniteDifferenceLog &log)
{
    log.clear();

    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = (int)curPos.rows();

    Eigen::MatrixXd testpos = curPos;
    testpos.setRandom();
    Eigen::VectorXd testedge = edgeDOFs;
    testedge.setRandom();

    constexpr int nedgedofs = SFF::numExtraDOFs;

    std::vector<int> epsilons = {-2, -3, -4, -5, -6};
    for (auto epsilon : epsilons)
    {

        double pert = std::pow(10.0, epsilon);

        for (int face = 0; face < nfaces; face++)
        {
            Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> deriv;
            Eigen::Matrix<double, 18 + 3 * nedgedofs, 18 + 3 * nedgedofs> hess;
            double result = mat.bendingEnergy(mesh, testpos, testedge, thicknesses[face], abars[face], bbars[face], face, &deriv, &hess);

            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Eigen::MatrixXd fwdpertpos = testpos;
                    Eigen::MatrixXd backpertpos = testpos;
                    fwdpertpos(mesh.faceVertex(face, j), k) += pert;
                    backpertpos(mesh.faceVertex(face, j), k) -= pert;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> fwdpertderiv;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> backpertderiv;
                    double fwdnewresult = mat.bendingEnergy(mesh, fwdpertpos, testedge, thicknesses[face], abars[face], bbars[face], face, &fwdpertderiv, NULL);
                    double backnewresult = mat.bendingEnergy(mesh, backpertpos, testedge, thicknesses[face], abars[face], bbars[face], face, &backpertderiv, NULL);
                    double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                    if(verbose) std::cout << '(' << j << ", " << k << ") " << findiff << " " << deriv(0, 3 * j + k) << std::endl;
                    log.addEntry(pert, deriv(0, 3 * j + k), findiff);

                    Eigen::MatrixXd derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                    if (verbose)
                    {
                        std::cout << derivdiff << std::endl;
                        std::cout << "//" << std::endl;
                        std::cout << hess.row(3 * j + k) << std::endl << std::endl;
                    }
                    for (int l = 0; l > 18 + 3 * nedgedofs; l++)
                    {
                        log.addEntry(pert, hess(3 * j + k, l), derivdiff(0, l));
                    }
                }
    
                int oppidx = mesh.vertexOppositeFaceEdge(face, j);
                if (oppidx != -1)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        Eigen::MatrixXd fwdpertpos = testpos;
                        Eigen::MatrixXd backpertpos = testpos;
                        fwdpertpos(oppidx, k) += pert;
                        backpertpos(oppidx, k) -= pert;
                        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> fwdpertderiv;
                        Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> backpertderiv;
                        double fwdnewresult = mat.bendingEnergy(mesh, fwdpertpos, testedge, thicknesses[face], abars[face], bbars[face], face, &fwdpertderiv, NULL);
                        double backnewresult = mat.bendingEnergy(mesh, backpertpos, testedge, thicknesses[face], abars[face], bbars[face], face, &backpertderiv, NULL);
                        double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                        Eigen::MatrixXd derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                        if (verbose)
                        {
                            std::cout << "opp (" << j << ", " << k << ") " << findiff << " " << deriv(0, 9 + 3 * j + k) << std::endl;
                            std::cout << derivdiff << std::endl;
                            std::cout << "//" << std::endl;
                            std::cout << hess.row(9 + 3 * j + k) << std::endl << std::endl;
                        }
                        log.addEntry(pert, deriv(0, 9 + 3 * j + k), findiff);
                        for (int l = 0; l < 18 + 3 * nedgedofs; l++)
                        {
                            log.addEntry(pert, hess(9 + 3 * j + k, l), derivdiff(0, l));
                        }                        
                    }
                }
                
                for (int k = 0; k < nedgedofs; k++)
                {
                    Eigen::VectorXd fwdpertedge = testedge;
                    Eigen::VectorXd backpertedge = testedge;
                    fwdpertedge[nedgedofs * mesh.faceEdge(face, j) + k] += pert;
                    backpertedge[nedgedofs * mesh.faceEdge(face, j) + k] -= pert;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> fwdpertderiv;
                    Eigen::Matrix<double, 1, 18 + 3 * nedgedofs> backpertderiv;
                    double fwdnewresult = mat.bendingEnergy(mesh, testpos, fwdpertedge, thicknesses[face], abars[face], bbars[face], face, &fwdpertderiv, NULL);
                    double backnewresult = mat.bendingEnergy(mesh, testpos, backpertedge, thicknesses[face], abars[face], bbars[face], face, &backpertderiv, NULL);
                    double findiff = (fwdnewresult - backnewresult) / 2.0 / pert;
                    if(verbose) std::cout << findiff << " " << deriv(0, 18 + nedgedofs * j + k) << std::endl;
                    log.addEntry(pert, deriv(0, 18 + nedgedofs * j + k), findiff);
                    
                    Eigen::MatrixXd derivdiff = (fwdpertderiv - backpertderiv) / 2.0 / pert;
                    if (verbose)
                    {
                        std::cout << derivdiff << std::endl;
                        std::cout << "//" << std::endl;
                        std::cout << hess.row(18 + nedgedofs * j + k) << std::endl << std::endl;
                    }
                    for (int l = 0; l < 18 + 3 * nedgedofs; l++)
                    {
                        log.addEntry(pert, hess(18 + nedgedofs * j + k, l), derivdiff(0, l));
                    }
                }
            }
        }
    }
}
