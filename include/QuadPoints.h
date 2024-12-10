//
// Created by Zhen Chen on 12/8/24.
//

#pragma once
#include <vector>
#include <Eigen/Core>

namespace LibShell {
struct QuadraturePoint
{
    double u;
    double v;
    double weight;
};

// this is based one the paper: http://lsec.cc.ac.cn/~tcui/myinfo/paper/quad.pdf and the corresponding source codes: http://lsec.cc.ac.cn/phg/download.htm (quad.c)
std::vector<QuadraturePoint> build_quadrature_points(int n);

}
