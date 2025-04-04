#include "HalfCylinder.h"
#include "igl/triangle/triangulate.h"
#include <sstream>
#include <cassert>
#include <iomanip>
#include "igl/remove_unreferenced.h"
#include "igl/boundary_loop.h"
#include <unordered_set>
void makeCylinder(bool regular,
                  double radius,
                  double height,
                  double triangleArea,
                  Eigen::MatrixXd& flatV,
                  Eigen::MatrixXd& V,
                  Eigen::MatrixXi& F,
                  double angle) {
    if (angle != 2 * M_PI) {
        angle = fmod(angle, 2 * M_PI);
    }
    double targetlength = 2.0 * std::sqrt(triangleArea / std::sqrt(3.0));

    int W = std::max(1, int(angle * radius / targetlength));
    int H = std::max(1, int(height / targetlength));

    if (regular) {
        flatV.resize((W + 1) * (H + 1), 3);
        V.resize((W + 1) * (H + 1), 3);
        F.resize(2 * W * H, 3);
        int curface = 0;
        for (int i = 0; i <= H; i++) {
            for (int j = 0; j <= W; j++) {
                int idx = i * (W + 1) + j;
                flatV(idx, 0) = double(j) / double(W) * angle * radius;
                flatV(idx, 1) = double(i) / double(H) * height;
                flatV(idx, 2) = 0;
                V(idx, 0) = radius * std::cos(flatV(idx, 0) / radius);
                V(idx, 1) = radius * std::sin(flatV(idx, 0) / radius);
                V(idx, 2) = flatV(idx, 1);
                if (i > 0 && j > 0) {
                    if ((curface / 2) % 2 == 0) {
                        int idxm1m1 = (i - 1) * (W + 1) + (j - 1);
                        int idxm1m0 = (i - 1) * (W + 1) + j;
                        F(curface, 0) = idxm1m1;
                        F(curface, 1) = idxm1m0;
                        F(curface, 2) = idx;
                        int idxm0m1 = i * (W + 1) + (j - 1);
                        F(curface + 1, 0) = idxm1m1;
                        F(curface + 1, 1) = idx;
                        F(curface + 1, 2) = idxm0m1;
                    } else {
                        int idxm1m1 = (i - 1) * (W + 1) + (j - 1);
                        int idxm1m0 = (i - 1) * (W + 1) + j;
                        int idxm0m1 = i * (W + 1) + (j - 1);

                        F(curface, 0) = idxm1m1;
                        F(curface, 1) = idxm1m0;
                        F(curface, 2) = idxm0m1;

                        F(curface + 1, 0) = idxm1m0;
                        F(curface + 1, 1) = idx;
                        F(curface + 1, 2) = idxm0m1;
                    }
                    curface += 2;
                }
            }
        }
    } else {
        Eigen::MatrixXd Vin(2 * W + 2 * H, 2);
        Eigen::MatrixXi E(2 * W + 2 * H, 2);
        Eigen::MatrixXd dummyH(0, 2);
        Eigen::MatrixXd V2;
        Eigen::MatrixXi F2;

        int vrow = 0;
        int erow = 0;
        // top boundary
        for (int i = 1; i < W; i++) {
            Vin(vrow, 0) = double(i) / double(W) * angle * radius;
            Vin(vrow, 1) = height;
            if (i > 1) {
                E(erow, 0) = vrow - 1;
                E(erow, 1) = vrow;
                erow++;
            }
            vrow++;
        }
        // bottom boundary
        for (int i = 1; i < W; i++) {
            Vin(vrow, 0) = double(i) / double(W) * angle * radius;
            Vin(vrow, 1) = 0;
            if (i > 1) {
                E(erow, 0) = vrow - 1;
                E(erow, 1) = vrow;
                erow++;
            }
            vrow++;
        }
        // left boundary
        for (int i = 0; i <= H; i++) {
            Vin(vrow, 0) = 0;
            Vin(vrow, 1) = double(i) / double(H) * height;
            if (i > 0) {
                E(erow, 0) = vrow - 1;
                E(erow, 1) = vrow;
                erow++;
            }
            vrow++;
        }
        // right boundary
        for (int i = 0; i <= H; i++) {
            Vin(vrow, 0) = angle * radius;
            Vin(vrow, 1) = double(i) / double(H) * height;
            if (i > 0) {
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

        for (int i = 0; i < nverts; i++) {
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

void makeTwistedCylinderWithoutSeam(bool regular,
                             double radius,
                             double height,
                             double triangleArea,
                             Eigen::MatrixXd& flatV,
                             Eigen::MatrixXi& flatF,
                             Eigen::MatrixXd& V,
                             Eigen::MatrixXi& F,
                             double twist_angle,
                             std::vector<int>* flatV_to_V_map) {
    double angle = 2 * M_PI;
    double targetlength = 2.0 * std::sqrt(triangleArea / std::sqrt(3.0));

    int W = std::max(1, int(angle * radius / targetlength));
    int H = std::max(1, int(height / targetlength));

    if (regular) {
        flatV.resize((W + 1) * (H + 1), 3);
        V.resize((W + 1) * (H + 1), 3);
        F.resize(2 * W * H, 3);
        int curface = 0;
        for (int i = 0; i <= H; i++) {
            for (int j = 0; j <= W; j++) {
                int idx = i * (W + 1) + j;
                flatV(idx, 0) = double(j) / double(W) * angle * radius;
                flatV(idx, 1) = double(i) / double(H) * height;
                flatV(idx, 2) = 0;
                
                V(idx, 0) = radius * std::cos(flatV(idx, 0) / radius);
                V(idx, 1) = radius * std::sin(flatV(idx, 0) / radius);
                V(idx, 2) = flatV(idx, 1);
                if (i > 0 && j > 0) {
                    if ((curface / 2) % 2 == 0) {
                        int idxm1m1 = (i - 1) * (W + 1) + (j - 1);
                        int idxm1m0 = (i - 1) * (W + 1) + j;
                        F(curface, 0) = idxm1m1;
                        F(curface, 1) = idxm1m0;
                        F(curface, 2) = idx;
                        int idxm0m1 = i * (W + 1) + (j - 1);
                        F(curface + 1, 0) = idxm1m1;
                        F(curface + 1, 1) = idx;
                        F(curface + 1, 2) = idxm0m1;
                    } else {
                        int idxm1m1 = (i - 1) * (W + 1) + (j - 1);
                        int idxm1m0 = (i - 1) * (W + 1) + j;
                        int idxm0m1 = i * (W + 1) + (j - 1);

                        F(curface, 0) = idxm1m1;
                        F(curface, 1) = idxm1m0;
                        F(curface, 2) = idxm0m1;

                        F(curface + 1, 0) = idxm1m0;
                        F(curface + 1, 1) = idx;
                        F(curface + 1, 2) = idxm0m1;
                    }
                    curface += 2;
                }
            }
        }
    } else {
        Eigen::MatrixXd Vin(2 * W + 2 * H, 2);
        Eigen::MatrixXi E(2 * W + 2 * H, 2);
        Eigen::MatrixXd dummyH(0, 2);
        Eigen::MatrixXd V2;
        Eigen::MatrixXi F2;

        int vrow = 0;
        int erow = 0;
        // top boundary
        for (int i = 1; i < W; i++) {
            Vin(vrow, 0) = double(i) / double(W) * angle * radius;
            Vin(vrow, 1) = height;
            if (i > 1) {
                E(erow, 0) = vrow - 1;
                E(erow, 1) = vrow;
                erow++;
            }
            vrow++;
        }
        // bottom boundary
        for (int i = 1; i < W; i++) {
            Vin(vrow, 0) = double(i) / double(W) * angle * radius;
            Vin(vrow, 1) = 0;
            if (i > 1) {
                E(erow, 0) = vrow - 1;
                E(erow, 1) = vrow;
                erow++;
            }
            vrow++;
        }
        // left boundary
        for (int i = 0; i <= H; i++) {
            Vin(vrow, 0) = 0;
            Vin(vrow, 1) = double(i) / double(H) * height;
            if (i > 0) {
                E(erow, 0) = vrow - 1;
                E(erow, 1) = vrow;
                erow++;
            }
            vrow++;
        }
        // right boundary
        for (int i = 0; i <= H; i++) {
            Vin(vrow, 0) = angle * radius;
            Vin(vrow, 1) = double(i) / double(H) * height;
            if (i > 0) {
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

        for (int i = 0; i < nverts; i++) {
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

    flatF = F;

    std::vector<int> left_boundary;
    std::vector<int> right_boundary;

    std::vector<int> bnd_loop;
    igl::boundary_loop(F, bnd_loop);

    for (int i = 0; i < bnd_loop.size(); i++) {
        if (std::abs(flatV(bnd_loop[i], 0)) < 1e-6) {
            left_boundary.push_back(bnd_loop[i]);
        } else if (std::abs(flatV(bnd_loop[i], 0) - angle * radius) < 1e-6) {
            right_boundary.push_back(bnd_loop[i]);
        }
    }

    std::vector<int> picked_left_boundary(left_boundary.size(), 0);
    std::vector<int> right_to_left_map;
    for (int j =  0; j < right_boundary.size(); j++) {
        for (int i = 0; i < left_boundary.size(); i++) {
            if (!picked_left_boundary[i]) {
                if (std::abs(flatV(left_boundary[i], 1) - flatV(right_boundary[j], 1)) < 1e-6) {
                    right_to_left_map.push_back(i);
                    picked_left_boundary[i] = 1;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < picked_left_boundary.size(); i++) {
        if (!picked_left_boundary[i]) {
            std::cout << "Warning: left boundary vertex " << left_boundary[i] << " not picked" << std::endl;
        }
    }

    Eigen::MatrixXd newV = V;
    std::unordered_set<int> right_boundary_set(right_boundary.begin(), right_boundary.end());
    std::vector<int> old_to_new_map(V.rows(), -1);
    // remove the right boundary vertices
    int vid = 0;
    for (int i = 0; i < V.rows(); i++) {
        if (right_boundary_set.count(i) == 0) {
            // not in the set
            newV.row(vid) = V.row(i);
            old_to_new_map[i] = vid;
            vid++;
        }
    }
    newV.conservativeResize(vid, 3);

    for (int i = 0; i < right_boundary.size(); i++) {
        int matched_left_boundary_idx = left_boundary[right_to_left_map[i]];
        int right_boundary_idx = right_boundary[i];
        old_to_new_map[right_boundary_idx] = old_to_new_map[matched_left_boundary_idx];
    }

    Eigen::MatrixXi newF(F.rows(), 3);
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            newF(i, j) = old_to_new_map[F(i, j)];
        }
    }

    V = std::move(newV);
    F = std::move(newF);
    if (flatV_to_V_map) {
        *flatV_to_V_map = std::move(old_to_new_map);
    }

    // twist the cylinder
    Eigen::MatrixXd twistedV = V;
    for (int i = 0; i < V.rows(); i++) {
        double h = V(i, 2);
        double rotation_angle = twist_angle * h / height;
        double x = V(i, 0);
        double y = V(i, 1);
        twistedV(i, 0) = x * std::cos(rotation_angle) - y * std::sin(rotation_angle);
        twistedV(i, 1) = x * std::sin(rotation_angle) + y * std::cos(rotation_angle);
        twistedV(i, 2) = h;
    }
    V = std::move(twistedV);
}

void getBoundaries(const Eigen::MatrixXi& F, std::vector<int>& bdryVertices) {
    std::vector<std::vector<int>> boundaries;
    igl::boundary_loop(F, boundaries);
    assert(boundaries.size() == 1);

    bdryVertices = boundaries[0];
}