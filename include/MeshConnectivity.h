#ifndef MESHCONNECTIVITY_H
#define MESHCONNECTIVITY_H

#include <Eigen/Core>

namespace LibShell {

    class MeshConnectivity
    {
    public:
        MeshConnectivity();
        MeshConnectivity(const Eigen::MatrixXi& F);

        int nFaces() const { return (int)F.rows(); }
        int nEdges() const { return (int)EV.rows(); }
        int faceVertex(int face, int vertidx) const { return F(face, vertidx); }
        int faceEdge(int face, int vertidx) const { return FE(face, vertidx); } // opposite vertex vertidx
        int faceEdgeOrientation(int face, int vertidx) const { return FEorient(face, vertidx); } // face = edgeFace(faceEdge(face, i), faceEdgeOrientation(face, i))

        int edgeVertex(int edge, int vertidx) const { return EV(edge, vertidx); }
        int edgeFace(int edge, int faceidx) const { return EF(edge, faceidx); }
        int edgeOppositeVertex(int edge, int faceidx) const { return EOpp(edge, faceidx); }
        int vertexOppositeFaceEdge(int face, int vertidx) const;

        const Eigen::MatrixXi& faces() const { return F; }

        //int oppositeFace(int face, int vertidx) const;

    private:
        int oppositeVertexIndex(int edge, int faceidx) const; // index i so that F(edgeFace(edge, faceidx), i) is *not* part of the edge
        int oppositeVertex(int edge, int faceidx) const;




    private:
        Eigen::MatrixXi F;
        Eigen::MatrixXi FE;
        Eigen::MatrixXi FEorient;
        Eigen::MatrixXi EV;
        Eigen::MatrixXi EF;
        Eigen::MatrixXi EOpp;
    };
};

#endif