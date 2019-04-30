#include "../include/MeshConnectivity.h"

#include <vector>
#include <map>

MeshConnectivity::MeshConnectivity()
{
    F.resize(0, 3);
    FE.resize(0, 3);
    FEorient.resize(0, 3);
    EV.resize(0, 2);
    EF.resize(0, 2);
    EOpp.resize(0, 2);
}

MeshConnectivity::MeshConnectivity(const Eigen::MatrixXi &F) : F(F)
{
    std::map<std::pair<int, int>, Eigen::Vector2i > edgeFaces;
    int nfaces = (int)F.rows();
    for(int i=0; i<nfaces; i++)
    {
        for(int j=0; j<3; j++)
        {
            int v0 = F(i, (j+1)%3);
            int v1 = F(i, (j+2)%3);
            int idx=0;
            if(v0 > v1) 
            {
                std::swap(v0,v1);
                idx = 1;
            }
            
            std::pair<int, int> p(v0,v1);
            auto it = edgeFaces.find(p);
            if(it == edgeFaces.end())
            {
                edgeFaces[p][idx] = i;
                edgeFaces[p][1-idx] = -1;
            }
            else
            {
                edgeFaces[p][idx] = i;
            }
        }
    }
    
    int nedges = (int)edgeFaces.size();
    FE.resize(nfaces, 3);
    FEorient.resize(nfaces, 3);
    EV.resize(nedges, 2);
    EF.resize(nedges, 2);
    EOpp.resize(nedges, 2);
    std::map<std::pair<int, int>, int> edgeIndices;
    
    int idx=0;
    for(auto it : edgeFaces)
    {
        edgeIndices[it.first] = idx;
        EV(idx, 0) = it.first.first;
        EV(idx, 1) = it.first.second;
        EF(idx, 0) = it.second[0];
        EF(idx, 1) = it.second[1];
        idx++;
    }
    
    for(int i=0; i<nfaces; i++)
    {
        for(int j=0; j<3; j++)
        {
            int v0 = F(i, (j+1)%3);
            int v1 = F(i, (j+2)%3);
            if(v0 > v1) std::swap(v0,v1);
            FE(i,j) = edgeIndices[std::pair<int,int>(v0,v1)];
        }
    }
    
    for (int i = 0; i < nedges; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            EOpp(i, j) = oppositeVertex(i, j);
        }
    }

    for (int i = 0; i < nfaces; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int edge = faceEdge(i, j);
            if (edgeFace(edge, 0) == i)
                FEorient(i, j) = 0;
            else
                FEorient(i, j) = 1;
        }
    }
}

int MeshConnectivity::oppositeVertexIndex(int edge, int faceidx) const
{
    int face = edgeFace(edge, faceidx);
    if(face == -1)
        return -1;
        
    for(int j=0; j<3; j++)
    {
        if(F(face, j) != edgeVertex(edge, 0) && F(face, j) != edgeVertex(edge, 1))
            return j;
    }        

    // unreachable
    return -1;
}

int MeshConnectivity::oppositeVertex(int edge, int faceidx) const
{
    int face = edgeFace(edge, faceidx);
    int idx = oppositeVertexIndex(edge, faceidx);
    if(idx == -1)
        return -1;
    return F(face, idx);
}

int MeshConnectivity::vertexOppositeFaceEdge(int face, int vertidx) const
{
    int edge = faceEdge(face, vertidx);
    int edgeorient = faceEdgeOrientation(face, vertidx);
    return edgeOppositeVertex(edge, 1 - edgeorient);
}
