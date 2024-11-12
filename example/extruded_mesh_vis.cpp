#include "../Optimization/include/NewtonDescent.h"

#include "../include/MeshConnectivity.h"
#include "../include/ElasticShell.h"
#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/MidedgeAngleSinFormulation.h"
#include "../include/MidedgeAverageFormulation.h"
#include "../include/MidedgeAngleThetaFormulation.h"
#include "../include/StVKMaterial.h"
#include "../include/TensionFieldStVKMaterial.h"
#include "../include/NeoHookeanMaterial.h"
#include "../include/RestState.h"
#include "../src/GeometryDerivatives.h"

#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>

#include <unordered_set>
#include <memory>
#include <filesystem>

#include <CLI/CLI.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <chrono>

int num_steps;
double grad_tol;
double f_tol;
double x_tol;
bool is_swap;

double young;
double thickness;
double poisson;
int matid;
int sffid;
int proj_type;

Eigen::MatrixXd cur_pos;
LibShell::MeshConnectivity mesh;

std::string input_mesh = "";
std::string output_folder = "";

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif  // !M_PI

static double edge_theta(const LibShell::MeshConnectivity& mesh,
						 const Eigen::MatrixXd& curPos,
						 int edge,
						 Eigen::Matrix<double, 1, 12>* derivative,  // edgeVertex, then edgeOppositeVertex
						 Eigen::Matrix<double, 12, 12>* hessian) {
	if (derivative) derivative->setZero();
	if (hessian) hessian->setZero();
	int v0 = mesh.edgeVertex(edge, 0);
	int v1 = mesh.edgeVertex(edge, 1);
	int v2 = mesh.edgeOppositeVertex(edge, 0);
	int v3 = mesh.edgeOppositeVertex(edge, 1);
	if (v2 == -1 || v3 == -1) return 0;  // boundary edge

	Eigen::Vector3d q0 = curPos.row(v0);
	Eigen::Vector3d q1 = curPos.row(v1);
	Eigen::Vector3d q2 = curPos.row(v2);
	Eigen::Vector3d q3 = curPos.row(v3);

	Eigen::Vector3d n0 = (q0 - q2).cross(q1 - q2);
	Eigen::Vector3d n1 = (q1 - q3).cross(q0 - q3);
	Eigen::Vector3d axis = q1 - q0;
	Eigen::Matrix<double, 1, 9> angderiv;
	Eigen::Matrix<double, 9, 9> anghess;

	double theta = LibShell::angle(n0, n1, axis, (derivative || hessian) ? &angderiv : NULL, hessian ? &anghess : NULL);

	return theta;
}

void lame_parameters(double& alpha, double& beta) {
	alpha = young * poisson / (1.0 - poisson * poisson);
	beta = young / 2.0 / (1.0 + poisson);
}

Eigen::VectorXd get_offset_mesh(const Eigen::MatrixXd& V,
                                const LibShell::MeshConnectivity& mesh,
                                const Eigen::VectorXd& edge_DOFs,
                                Eigen::MatrixXd& offset_V,
                                Eigen::MatrixXi& offset_F,
                                double distance = 0.1) {
    offset_V.resize(mesh.nFaces() * 3, 3);
    Eigen::VectorXd err = Eigen::VectorXd::Zero(V.rows());
    offset_F.resize(mesh.nFaces(), 3);

    double ave_edge_len = 0;
    for (int eid = 0; eid < mesh.nEdges(); eid++) {
        ave_edge_len += (V.row(mesh.edgeVertex(eid, 0)) - V.row(mesh.edgeVertex(eid, 1))).norm();
    }
    ave_edge_len /= mesh.nEdges();

    double z = distance * ave_edge_len;
    std::vector<std::vector<int>> vmaps(V.rows());

    for (int fid = 0; fid < mesh.nFaces(); fid++) {
        std::vector<Eigen::Vector3d> es(3), es_perp(3);
        for (int i = 0; i < 3; i++) {
            es[i] = (V.row(mesh.faceVertex(fid, (i + 2) % 3)) - V.row(mesh.faceVertex(fid, (i + 1) % 3)));
        }
        Eigen::Vector3d face_normal = es[1].cross(es[2]).normalized();
        std::vector<double> edge_angles(3), zis(3);
        for (int i = 0; i < 3; i++) {
            es_perp[i] = es[i].cross(face_normal).normalized();
            int eid = mesh.faceEdge(fid, i);
            double angle = edge_theta(mesh, V, eid, nullptr, nullptr);
            double orient = mesh.faceEdgeOrientation(fid, i) == 0 ? 1.0 : -1.0;
            edge_angles[i] = 0.5 * angle + orient * edge_DOFs[eid];
            zis[i] = z * std::tan(edge_angles[i]);
        }

        for (int i = 0; i < 3; i++) {
            offset_V.row(fid * 3 + i) = z * face_normal + zis[(i + 1) % 3] * es_perp[(i + 1) % 3] +
                                        zis[(i + 2) % 3] * es_perp[(i + 2) % 3] - zis[i] * es_perp[i];
            offset_V.row(fid * 3 + i) += V.row(mesh.faceVertex(fid, i));
            vmaps[mesh.faceVertex(fid, i)].push_back(fid * 3 + i);
        }
        offset_F.row(fid) << 3 * fid, 3 * fid + 1, 3 * fid + 2;

		// check the correctness of the offset mesh
        for (int i = 0; i < 3; i++) {
            Eigen::Vector3d Mz_i =
                (offset_V.row(fid * 3 + (i + 1) % 3) + offset_V.row(fid * 3 + (i + 2) % 3)) / 2;
            Eigen::Vector3d Mz = V.row(mesh.faceVertex(fid, (i + 1) % 3)) + V.row(mesh.faceVertex(fid, (i + 2) % 3));
            Mz /= 2;

			Eigen::Vector3d mid_edge_normal = Mz_i - Mz;
            mid_edge_normal.normalize();

			double is_perp = mid_edge_normal.dot(es[i]);

			Eigen::Vector3d tmp = (offset_V.row(fid * 3 + i) - V.row(mesh.faceVertex(fid, i)));
            double dist = tmp.dot(face_normal);

            if (std::abs(is_perp) > 1e-6) {
                spdlog::error("The offset mesh is not perpendicular to the edge");
            }

			if (std::abs(dist - z) > 1e-6) {
                spdlog::error("The offset mesh is not at the correct distance, expected: {}, actual: {}", z, dist);
			}

			if (std::abs((Mz_i - Mz).norm() - z / std::cos(edge_angles[i])) > 1e-6) {
                spdlog::error("Incorrect midpoints distance, expected: {}, actual: {}", z / std::cos(edge_angles[i]),
                              (Mz_i - Mz).norm());
            }
        }
    }

	for (int i = 0; i < vmaps.size(); i++) {
        if (vmaps[i].size() > 1) {
            for (int j = 0; j < vmaps[i].size(); j++) {
                for (int k = j + 1; k < vmaps[i].size(); k++) {
                    err[i] += (offset_V.row(vmaps[i][j]) - offset_V.row(vmaps[i][k])).squaredNorm();
				}
			}
        }
    }

    return err;
  
}

template <class SFF>
double run_simulation(const LibShell::MeshConnectivity& mesh,
					  const Eigen::MatrixXd& rest_pos,
					  const Eigen::VectorXd& rest_edge_DOFs,
					  Eigen::MatrixXd& cur_pos,
					  Eigen::VectorXd& cur_edge_DOFs,
					  const std::unordered_set<int>* fixed_verts,
					  double thickness,
					  double lame_alpha,
					  double lame_beta,
					  int matid,
					  int proj_type) {
	// initialize default edge DOFs (edge director angles)
	if (cur_edge_DOFs.size() != SFF::numExtraDOFs * mesh.nEdges()) {
		spdlog::error("Invalid edge DOFs size");
		return 0;
	}

	// initialize the rest geometry of the shell
	LibShell::MonolayerRestState rest_state;

	// set uniform thicknesses
	rest_state.thicknesses.resize(mesh.nFaces(), thickness);

	// initialize first fundamental forms to those of input mesh
	LibShell::ElasticShell<SFF>::firstFundamentalForms(mesh, rest_pos, rest_state.abars);

	// initialize second fundamental forms to those of input mesh
	rest_state.bbars.resize(mesh.nFaces());
	for (int i = 0; i < mesh.nFaces(); i++) {
		rest_state.bbars[i].setZero();
	}

	rest_state.lameAlpha.resize(mesh.nFaces(), lame_alpha);
	rest_state.lameBeta.resize(mesh.nFaces(), lame_beta);

	std::shared_ptr<LibShell::MaterialModel<SFF>> mat;
	switch (matid) {
		case 0:
			mat = std::make_shared<LibShell::NeoHookeanMaterial<SFF>>();
			break;
		case 1:
			mat = std::make_shared<LibShell::StVKMaterial<SFF>>();
			break;
		case 2:
			mat = std::make_shared<LibShell::TensionFieldStVKMaterial<SFF>>();
			break;
		default:
			assert(false);
	}

	// projection matrix
	Eigen::SparseMatrix<double> P;
	std::vector<Eigen::Triplet<double>> Pcoeffs;
	int nedges = mesh.nEdges();
	int nedgedofs = SFF::numExtraDOFs;
	// we only allow fixed vertices in the current implementation
	Eigen::VectorXd fixed_dofs(3 * cur_pos.rows());
	fixed_dofs.setZero();
	int nfree = 0;
	for (int i = 0; i < cur_pos.rows(); i++) {
		if (!fixed_verts || !fixed_verts->count(i)) {
			Pcoeffs.push_back({nfree, 3 * i, 1.0});
			Pcoeffs.push_back({nfree + 1, 3 * i + 1, 1.0});
			Pcoeffs.push_back({nfree + 2, 3 * i + 2, 1.0});
			nfree += 3;
		} else {
			fixed_dofs.segment<3>(3 * i) = cur_pos.row(i).transpose();
		}
	}
	for (int i = 0; i < nedges * nedgedofs; i++) {
		Pcoeffs.push_back(Eigen::Triplet<double>(nfree, 3 * cur_pos.rows() + i, 1.0));
		nfree++;
	}

	P.resize(nfree, 3 * cur_pos.rows() + nedges * nedgedofs);
	P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());

	int totalDOFs = 3 * cur_pos.rows() + nedges * nedgedofs;

	// project the current position
	auto pos_edgedofs_to_variable = [&](const Eigen::MatrixXd& pos, const Eigen::VectorXd& edge_DOFs) {
		Eigen::VectorXd var(nfree);
		int n = 0;
		for (int i = 0; i < pos.rows(); i++) {
			if (!fixed_verts || !fixed_verts->count(i)) {
				var.segment<3>(n) = pos.row(i).transpose();
				n += 3;
			}
		}
		var.tail(nedges * nedgedofs) = edge_DOFs;
		return var;
	};

	auto variable_to_pos_edgedofs = [&](const Eigen::VectorXd& var) {
		Eigen::MatrixXd pos(cur_pos.rows(), 3);
		int n = 0;
		for (int i = 0; i < cur_pos.rows(); i++) {
			if (!fixed_verts || !fixed_verts->count(i)) {
				pos.row(i) = var.segment<3>(n).transpose();
				n += 3;
			} else {
				pos.row(i) = fixed_dofs.segment<3>(3 * i).transpose();
			}
		}
		Eigen::VectorXd edge_DOFs = var.tail(nedges * nedgedofs);
		return std::pair<Eigen::MatrixXd, Eigen::VectorXd>{pos, edge_DOFs};
	};

	// energy, gradient, and hessian
	auto obj_func = [&](const Eigen::VectorXd& var, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hessian,
						bool psd_proj) {
		Eigen::MatrixXd pos;
		Eigen::VectorXd edge_DOFs;
		std::vector<Eigen::Triplet<double>> hessian_triplets;
		std::tie(pos, edge_DOFs) = variable_to_pos_edgedofs(var);

		double energy =
			LibShell::ElasticShell<SFF>::elasticEnergy(mesh, pos, edge_DOFs, *mat, rest_state, psd_proj ? proj_type : 0,
													   grad, hessian ? &hessian_triplets : nullptr);

		if (grad) {
			if (fixed_verts) {
				*grad = P * (*grad);
			}
		}

		if (hessian) {
			hessian->resize(totalDOFs, totalDOFs);
			hessian->setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
			if (fixed_verts) {
				*hessian = P * (*hessian) * P.transpose();
			}
		}

		return energy;
	};

	auto find_max_step = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) { return 1.0; };

	Eigen::VectorXd x0 = pos_edgedofs_to_variable(cur_pos, cur_edge_DOFs);

	if (output_folder != "") {
		auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_folder + "/log.txt", true);
		spdlog::flush_every(std::chrono::seconds(1));

		auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

		auto multi_sink_logger =
			std::make_shared<spdlog::logger>("Newton Solver", spdlog::sinks_init_list{file_sink, console_sink});

		spdlog::set_default_logger(multi_sink_logger);
	}

	double init_energy = obj_func(x0, nullptr, nullptr, false);

	OptSolver::TestFuncGradHessian(obj_func, x0);

	double init_norm = cur_edge_DOFs.norm();

	OptSolver::NewtonSolver(obj_func, find_max_step, x0, num_steps, grad_tol, x_tol, f_tol, proj_type != 0, true,
							is_swap);

	std::tie(cur_pos, cur_edge_DOFs) = variable_to_pos_edgedofs(x0);

	double energy = obj_func(x0, nullptr, nullptr, false);


	spdlog::info("Initial Energy: {}, Optimized Energy: {}", init_energy, energy);
    spdlog::info("Initial edge dofs norm: {}, Optimized edge dofs norm: {}", init_norm, cur_edge_DOFs.norm());

	return obj_func(x0, nullptr, nullptr, false);
}


int main(int argc, char* argv[]) {
	CLI::App app("Visualize the Tan-formula Extruded Mesh");
    app.add_option("Input, -i, --input", input_mesh, "Input Mesh")->check(CLI::ExistingFile)->required(true);

	// optimization parameters
	app.add_option("--num-steps", num_steps, "Number of iteration")->default_val(30);
	app.add_option("--grad-tol", grad_tol, "Gradient tolerance")->default_val(1e-6);
	app.add_option("--f-tol", f_tol, "Function tolerance")->default_val(0);
	app.add_option("--x-tol", x_tol, "Variable tolerance")->default_val(0);

	// material parameters
	app.add_option("--young", young, "Young's Modulus")->default_val(1e9);
	app.add_option("--thickness", thickness, "Thickness")->default_val(1e-4);
	app.add_option("--poisson", poisson, "Poisson's Ratio")->default_val(0.5);
	app.add_option("--material", matid, "Material Model")->default_val(1);
	app.add_option("--projection", proj_type, "Hessian Projection Type, 0 : no projection, 1: max(H, 0), 2: Abs(H)")
		->default_val(1);
	app.add_flag("--swap", is_swap, "Swap to Actual Hessian when close to optimum");

	app.add_option("ouput,-o,--output", output_folder, "Output folder");
	CLI11_PARSE(app, argc, argv);

	// make output folder
	if (output_folder != "") {
		std::filesystem::create_directories(output_folder);
	}

	// generate mesh
	Eigen::MatrixXd orig_V, rest_V;
	Eigen::MatrixXi F;

	igl::readOBJ(input_mesh, orig_V, F);
    rest_V = orig_V;

	double distance = 1;

	std::unordered_set<int> fixed_verts;
	for (int i = 0; i < rest_V.rows(); i++) {
		fixed_verts.insert(i);
	}

	// set up mesh connectivity
	mesh = LibShell::MeshConnectivity(F);

	// initial position
	cur_pos = orig_V;

	Eigen::VectorXd rest_edge_DOFs, cur_edge_DOFs, init_edge_DOFs;
	rest_edge_DOFs = Eigen::VectorXd::Zero(mesh.nEdges());
	cur_edge_DOFs = rest_edge_DOFs;
	init_edge_DOFs = rest_edge_DOFs;

	double energy = 0;


	polyscope::init();

	// Register a surface mesh structure
	auto surface_mesh = polyscope::registerSurfaceMesh("Rest mesh", rest_V, F);
	surface_mesh->setEnabled(false);

	auto cur_surface_mesh = polyscope::registerSurfaceMesh("Current mesh", cur_pos, F);

	polyscope::state::userCallback = [&]() {
		if (ImGui::Button("Reset", ImVec2(-1, 0))) {
			cur_pos = orig_V;
			cur_surface_mesh->updateVertexPositions(cur_pos);
		}

		if (ImGui::CollapsingHeader("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputDouble("Thickness", &thickness);
			ImGui::InputDouble("Poisson's Ration", &poisson);
			ImGui::Combo("Material Model", &matid, "NeoHookean\0StVK\0\0");
		}

		if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Combo("Hessian Projection", &proj_type, "No Projection\0Max Zero\0Abs\0\0");
			ImGui::InputInt("Num Steps", &num_steps);
			ImGui::InputDouble("Gradient Tol", &grad_tol);
			ImGui::InputDouble("Function Tol", &f_tol);
			ImGui::InputDouble("Variable Tol", &x_tol);
			ImGui::Checkbox("Swap to Actual Hessian when close to optimum", &is_swap);

			if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0))) {
				double lame_alpha, lame_beta;
				lame_parameters(lame_alpha, lame_beta);

			   switch (sffid) {
					case 0:
						energy = run_simulation<LibShell::MidedgeAngleTanFormulation>(
							mesh, rest_V, rest_edge_DOFs, cur_pos, cur_edge_DOFs, &fixed_verts, thickness, lame_alpha,
							lame_beta, matid, proj_type);
						break;
					case 1:
						energy = run_simulation<LibShell::MidedgeAngleSinFormulation>(
							mesh, rest_V, rest_edge_DOFs, cur_pos, cur_edge_DOFs, &fixed_verts, thickness, lame_alpha,
							lame_beta, matid, proj_type);
						break;
					case 2:
						energy = run_simulation<LibShell::MidedgeAngleThetaFormulation>(
							mesh, rest_V, rest_edge_DOFs, cur_pos, cur_edge_DOFs, &fixed_verts, thickness, lame_alpha,
							lame_beta, matid, proj_type);
						break;
					default:
						assert(false);
				}
				cur_surface_mesh->updateVertexPositions(cur_pos);
				if (output_folder != "") {
					igl::writeOBJ(output_folder + "/deformed.obj", cur_pos, F);
				}
			}
		}

		ImGui::InputDouble("Relative Offset Distance", &distance);

		if (ImGui::Button("Compute the offset mesh", ImVec2(-1, 0))) {
			if (init_edge_DOFs.size() && cur_edge_DOFs.size()) {
				Eigen::MatrixXd offset_V;
				Eigen::MatrixXi offset_F;
				Eigen::VectorXd init_err = get_offset_mesh(cur_pos, mesh, init_edge_DOFs, offset_V, offset_F, distance);
				auto offset_mesh = polyscope::registerSurfaceMesh("Init Offset mesh", offset_V, offset_F);
				Eigen::VectorXd cur_err = get_offset_mesh(cur_pos, mesh, cur_edge_DOFs, offset_V, offset_F, distance);
				auto cur_offset_mesh = polyscope::registerSurfaceMesh("Current Offset mesh", offset_V, offset_F);

				cur_surface_mesh->addVertexScalarQuantity("Initial Error", init_err);
                cur_surface_mesh->addVertexScalarQuantity("Current Error", cur_err);

				spdlog::info("Initial Error: {}, Current Error: {}", init_err.sum(), cur_err.sum());
			}
			
		}
	};

	// View the point cloud and mesh we just registered in the 3D UI
	polyscope::show();
}
