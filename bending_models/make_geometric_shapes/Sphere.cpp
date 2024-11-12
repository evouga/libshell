#define _USE_MATH_DEFINES
#include "Sphere.h"
#include <random>
#include "../quickhull//QuickHull.hpp"

void makeSphere(double radius, double triangleArea,Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    int verts = 2.0 * M_PI * radius * radius / triangleArea;
    verts = std::max(verts, 4);
    sphereFromSamples(verts, V, F);
    V *= radius;
}

void sphereFromSamples(int samples, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    std::default_random_engine generator;
    std::normal_distribution<> nd(0.0, 1.0);
    std::vector<quickhull::Vector3<double> > pts(samples);
    for(int i=0; i<samples; i++)
    {
        Eigen::Vector3d v;
        for(int j=0; j<3; j++)
            v[j] = nd(generator);
        v /= v.norm();
        pts[i].x = v[0];
        pts[i].y = v[1];
        pts[i].z = v[2];
    }
    
    bool flag;
    quickhull::QuickHull<double> qh;
    quickhull::ConvexHull<double> hull = qh.getConvexHull(pts, true, false, flag);
    
    int verts = hull.getVertexBuffer().size();
    V.resize(verts, 3);
    for(int i=0; i<verts; i++)
    {
        V(i,0) = hull.getVertexBuffer()[i].x;
        V(i,1) = hull.getVertexBuffer()[i].y;
        V(i,2) = hull.getVertexBuffer()[i].z;
    }
    
    int faces = hull.getIndexBuffer().size()/3;
    F.resize(faces, 3);
    for(int i=0; i<faces; i++)
    {
        for(int j=0; j<3; j++)
        {
            F(i, j) = hull.getIndexBuffer()[3*i+j];
        }
    }
}
