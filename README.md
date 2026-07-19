# LibShell

This code implements thin shell energies and their derivatives and Hessians.

## Kinematics

The shell's current pose is represented as a triangle mesh, as well as, optionally, some per-edge normal directors. The rest state is specified with per-face first and second fundamental forms; these can be computed from the current pose, or from a separate mesh (with identical combinatorics) in a rest pose, or procedurally specified, etc.

## Bending Options

Four options are implemented for how the bending energy is discretized.

* MidedgeAngleSinFormulation: bending energy is roughly sin(theta/2) for edge turning angle theta. This bending energy is the DCS energy (without the shear one-form term) described in Weischedel, _A Discrete Geometric View on Shear-deformable Shell Models_, 2012.

* MidedgeAngleTanFormulation: energy is roughly tan(theta/2) instead. The main difference of this formulation from the previous one is that the bending energy diverges for 180-degree bent hinges. This is the BAC model described in Chen et al., _Better Bending: Analysis, Construction and Verification of Discrete Bending Models for Kirchhoff-Love Shells_, SIGGRAPH 2026.

* MidedgeAngleThetaFormulation: the Discrete Shape Operator-based bending energy described in Grinspun et al., _Computing Discrete Shape Operators on General Meshes_, CGF, 2006.

* MidedgeAverageFormulation: eschews the normal directors of the models and instead assumes that the normal direction on an edge is always the mean of the neighboring face normals, as described in Chen et al., _Physical Simulation of Environmentally Induced Thin Shell Deformation_, SIGGRAPH 2018.

All else being equal, and especially if simulating surfaces that might undergo large curvature in regions under-resolved by the discrete mesh, I recommend the `MidedgeAngleTanFormulation` as its energy barrier will prevent hinges from collapsing into the 180-degree bent state in those regions. For detailed experiments please see the _Better Bending_ paper cited above.

## Material Model

Both a St. Venant-Kirchhoff and Neo-Hookean material model are implemented (and can be chosen independently of the above options for how to discretize shell bending energy). Pass the desired `MaterialModel` to the elastic energy computation. Each material model assumes uniform Lamé parameters over the entire surface (but you can specify different thicknesses for each triangle). 

For the St. Venant-Kirchhoff material, there is a bilayer implementation (where each half of the shell has a different thickness, Lamé constants, and strain-free state). See `BilayerStVKMaterial`.

Also implemented is a tension-field version of the St. Venant-Kirchhoff material. This material resists tension only (and not compression or bending).


See the example program for the formulas that convert Young's modulus and Poisson's ratio to Lamé parameters. Note that the 2D formulas are *not* the same as the 3D ones found on e.g. Wikipedia.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This procedure will build:
 - the library itself;
 - an example program, which performs a few iterations of a static solve on an rest-flat bunny mesh;
 - a testing program, used to verify the correctness of the energies and derivatives.

## Dependencies

The library itself depends only on Eigen (set the environment variable `EIGEN3_INCLUDE_DIR` to point to your Eigen folder). The example program includes a viewer which uses polyscope, and libigl for mesh io.

## Compiling on Windows

Due to poor interoperation of the Eigen library with the MSVC compiler, Release mode compilation of the derivative code on Windows can take forever (over 8 hours). The issue appears to have been resolved as of the 2019 edition of MSVC. If you are having issues with compile time, to solve the problem add EIGEN_STRONG_INLINE=inline to your preprocessor macros when building libshell.

## Testing Program

I've included code in tests/ that performs sanity-checking on the shell energy implementation. In particular, the program performs and reports information on the following tests:
1. All implemented analytic derivatives and Hessians are checked against the corresponding energy and derivative (respectively) using centered finite differences.
2. All (consitutive model, second fundamental form) pairs are checked against each other for consistency in the infinitesimal-strain regime about the flat rest state (i.e. that their Hessians all agree at this point).
3. The single-layer and bilayer implementations of the St. Venant-Kirchhoff material are compared against each other for consistency (the monolayer should be exactly equivalent to the bilayer, when both bilayers have identical parameters and rest state).