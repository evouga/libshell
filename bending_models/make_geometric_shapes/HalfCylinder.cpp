#include "HalfCylinder.h"
#include "igl/triangle/triangulate.h"
#include <sstream>
#include <cassert>
#include <iomanip>
#include "igl/remove_unreferenced.h"
#include "igl/boundary_loop.h"

void makeHalfCylinder(bool regular, double radius, double height, double triangleArea,
    Eigen::MatrixXd& flatV,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F)
{
    constexpr double PI = 3.1415926535898;
    double targetlength = 2.0 * std::sqrt(triangleArea / std::sqrt(3.0));

    int W = std::max(1, int(PI * radius / targetlength));
    int H = std::max(1, int(height / targetlength));

    if (regular)
    {
        flatV.resize(W * H, 3);
        V.resize(W * H, 3);
        F.resize(2 * (W - 1) * (H - 1), 3);
        int curface = 0;
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                int idx = i * W + j;
                flatV(idx, 0) = double(j) / double(W) * PI * radius;
                flatV(idx, 1) = double(i) / double(H) * height;
                flatV(idx, 2) = 0;
                V(idx, 0) = radius * std::cos(flatV(idx,0) / radius);
                V(idx, 1) = radius * std::sin(flatV(idx,0) / radius);
                V(idx, 2) = flatV(idx, 1);
                if (i > 0 && j > 0)
                {
                    int idxm1m1 = (i - 1) * W + (j - 1);
                    int idxm1m0 = (i - 1) * W + j;
                    F(curface, 0) = idxm1m1;
                    F(curface, 1) = idxm1m0;
                    F(curface, 2) = idx;
                    int idxm0m1 = i * W + (j - 1);
                    F(curface + 1, 0) = idxm1m1;
                    F(curface + 1, 1) = idx;
                    F(curface + 1, 2) = idxm0m1;
                    curface += 2;
                }
            }
        }
    }
    else
    {
        Eigen::MatrixXd Vin(2 * W + 2 * H, 2);
        Eigen::MatrixXi E(2 * W + 2 * H, 2);
        Eigen::MatrixXd dummyH(0, 2);
        Eigen::MatrixXd V2;
        Eigen::MatrixXi F2;

        int vrow = 0;
        int erow = 0;
        // top boundary
        for (int i = 1; i < W; i++)
        {
            Vin(vrow, 0) = double(i) / double(W) * PI * radius;
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
            Vin(vrow, 0) = double(i) / double(W) * PI * radius;
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
            Vin(vrow, 0) = PI * radius;
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

        flatV.resize(nverts, 3);
        Eigen::MatrixXd rolledV(nverts, 3);

        for (int i = 0; i < nverts; i++)
        {
            Eigen::Vector2d q = V2.row(i).transpose();
            Eigen::Vector3d rolledq;
            flatV(i, 0) = q[0];
            flatV(i, 1) = q[1];
            flatV(i, 2) = 0;
            rolledq[0] = radius * std::cos(q[0] / radius);
            rolledq[1] = radius * std::sin(q[0] / radius);
            rolledq[2] = q[1];
            rolledV.row(i) = rolledq.transpose();
        }

        V = rolledV;
        F = F2;
    }
}

void getBoundaries(const Eigen::MatrixXi& F, std::vector<int>& bdryVertices)
{
    std::vector<std::vector<int> > boundaries;
    igl::boundary_loop(F, boundaries);
    assert(boundaries.size() == 1);

    bdryVertices = boundaries[0];
}