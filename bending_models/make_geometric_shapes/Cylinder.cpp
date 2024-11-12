#include "Cylinder.h"
#include "igl/triangle/triangulate.h"
#include <sstream>
#include <cassert>
#include <iomanip>
#include "igl/remove_unreferenced.h"
#include "igl/boundary_loop.h"

void makeCylinder(double radius, double height, double triangleArea,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F)
{
    constexpr double PI = 3.1415926535898;
    double targetlength = 2.0 * std::sqrt(triangleArea / std::sqrt(3.0));

    int W = std::max(1, int(2.0 * PI * radius / targetlength));
    int H = std::max(1, int(height / targetlength));
    Eigen::MatrixXd Vin(2*W + 2*H, 2);
    Eigen::MatrixXi E(2*W+2*H, 2);
    Eigen::MatrixXd dummyH(0, 2);
    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;

    int vrow = 0;
    int erow = 0;
    // top boundary
    for (int i = 1; i < W; i++)
    {
        Vin(vrow, 0) = double(i) / double(W) * 2.0 * PI * radius;
        Vin(vrow, 1) = height;
        if (i > 1)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // bottom boundary
    for (int i = 1; i < W; i++)
    {
        Vin(vrow, 0) = double(i) / double(W) * 2.0 * PI * radius;
        Vin(vrow, 1) = 0;
        if (i > 1)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // left boundary    
    for (int i = 0; i <= H; i++)
    {
        Vin(vrow, 0) = 0;
        Vin(vrow, 1) = double(i) / double(H) * height;        
        if (i > 0)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // right boundary    
    for (int i = 0; i <= H; i++)
    {
        Vin(vrow, 0) = 2.0 * PI * radius;
        Vin(vrow, 1) = double(i) / double(H) * height;
        if (i > 0)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // missing four edges
    E(erow, 0) = (W - 1) - 1;
    E(erow, 1) = 2 * (W - 1) + 2 * (H + 1) - 1;
    erow++;
    E(erow, 0) = 2 * (W - 1) + (H + 1);
    E(erow, 1) = 2 * (W - 1) - 1;
    erow++;
    E(erow, 0) = W - 1;
    E(erow, 1) = 2 * (W - 1);
    erow++;
    E(erow, 0) = 2 * (W - 1) + (H + 1) - 1;
    E(erow, 1) = 0;
    erow++;

    assert(vrow == 2 * H + 2 * W);
    assert(erow == 2 * H + 2 * W);
    std::stringstream ss;
    ss << "a" << std::setprecision(30) << std::fixed << triangleArea << "qDY";
    igl::triangle::triangulate(Vin, E, dummyH, ss.str(), V2, F2);
    
    // roll up

    int nverts = V2.rows();

    Eigen::MatrixXd rolledV(nverts, 3);

    for (int i = 0; i < nverts; i++)
    {
        Eigen::Vector2d q = V2.row(i).transpose();
        Eigen::Vector3d rolledq;
        rolledq[0] = radius * std::cos(q[0] / radius);
        rolledq[1] = radius * std::sin(q[0] / radius);
        rolledq[2] = q[1];
        rolledV.row(i) = rolledq.transpose();
    }

    // fuse

    int nfaces = F2.rows();

    Eigen::MatrixXi rolledF(nfaces, 3);

    for (int i = 0; i < nfaces; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int idx = F2(i, j);
            if (idx >= 2 * (W - 1) + (H + 1) && idx < 2 * (W - 1) + 2 * (H + 1))
            {
                idx -= (H + 1);
            }
            rolledF(i, j) = idx;
        }
    }    
    Eigen::VectorXi I;
    igl::remove_unreferenced(rolledV, rolledF, V, F, I);
}

void getBoundaries(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::vector<int>& topVertices, std::vector<int>& bottomVertices)
{
    std::vector<std::vector<int> > boundaries;
    igl::boundary_loop(F, boundaries);
    assert(boundaries.size() == 2);

    if (V(boundaries[0][0], 2) < V(boundaries[1][0], 2))
    {
        topVertices = boundaries[1];
        bottomVertices = boundaries[0];
    }
    else
    {
        topVertices = boundaries[0];
        bottomVertices = boundaries[1];
    }
}