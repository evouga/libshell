#ifndef RESTSTATE_H
#define RESTSTATE_H

namespace LibShell {

    enum class RestStateType
    {
        RST_NONE,
        RST_MONOLAYER,
        RST_BILAYER
    };

    struct RestState
    {
    public:
        virtual RestStateType type() const { return RestStateType::RST_NONE; }
    };

    /* Encodes the rest state information for an elastic monolayer.
     * - thicknesses:   |F| x 1 list of triangle thicknesses.
     * - abars, bbars:  first and second fundamental forms, in the barycentric coordinates of each mesh face, encoding the shell rest state.
        If you have explicit rest geometry, you can compute these using the *FundamentalForms functions. Alternatively you
        can set the forms directly (zero matrices for bbar if you want a flat rest state, for instance).
     */
    struct MonolayerRestState : public RestState
    {
    public:
        virtual RestStateType type() const { return RestStateType::RST_MONOLAYER; }

        std::vector<double> thicknesses;
        std::vector<Eigen::Matrix2d> abars;
        std::vector<Eigen::Matrix2d> bbars;
    };

    /*
     * Encodes the rest state information for an elastic bilayer.
     * - layers:    contains the data for each of the two layers. Each layer is an elastic monolayer.
     * Note that a bilayer with layers[0] == layers[1] is identical to a monolayer with twice the thickness.
     */
    struct BilayerRestState : public RestState
    {
    public:
        virtual RestStateType type() const { return RestStateType::RST_BILAYER; }

        MonolayerRestState layers[2];
    };
};

#endif
