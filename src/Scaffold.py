""" SCAFFOLD MODULE. Will include methods to compute the scaffold as a result of
the filtration of minimal homology bases, and plot/perform analysis.
"""

import numpy as np
import Geometry
import networkx as nx

def frequency_Scaffold( Filtr , NV  ):
    """        Computes the symmetric, zero-diagonal matrix of the frequency
    scaffold based on a filtration of minimal homology bases.  TODO: it could
    take into account the distance between filtration steps,  making a  weighted
    scaffold """

    if type(NV) is not int or NV <= 0:
        raise ValueError("Number of vertices not correct")
    if type(Filtr) is not Geometry.Filtration:
        raise ValueError("Not a filtration")
    if not Filtr.edgesGiven():
        raise ValueError("Filtration must be given an EdgeList")

    S = np.zeros( (NV,NV) , dtype=float )

    edList = np.array(Filtr.get_edgeList()) # save the common edge list
    for eps,B in Filtr:
        vB = B.smear() # list of basis cycles, with draws smeared
        for vC in vB: # basis cycle, smeared (not 0-1)
            edges = edList[ vC > 0 ] # select the edges in the cycle
            weights = vC[ vC > 0 ] # select its "weight" too
            for i in range(len(edges)):
                x = edges[i][0]
                y = edges[i][1]
                w = weights[i]
                S[x,y] += w
                S[y,x] += w # keep symmetric!

    return S

def matrix_Basis(Bas , NV, EdgeList):
    """ Computes the symmetric, zero-diagonal adjacency matrix of a minimal
    homology basis.  Given its EdgeList and number of vertices. """
    if type(NV) is not int or NV <= 0:
        raise ValueError("Number of vertices not correct")
    if type(Bas) is not Geometry.Basis:
        raise ValueError("Not a filtration")

    M = np.zeros( (NV,NV) , dtype=float )
    EdgeList = np.array(EdgeList)

    vB = Bas.smear()
    for vC in vB: # basis cycle, smeared (not 0-1)
        edges = EdgeList[ vC > 0 ] # select the edges in the cycle
        weights = vC[ vC > 0 ] # select its "weight" too
        for i in range(len(edges)):
            x = edges[i][0]
            y = edges[i][1]
            w = weights[i]
            M[x,y] += w
            M[y,x] += w # keep symmetric!

    return M

def filtrationBases(Filtr , NV):
    if type(NV) is not int or NV <= 0:
        raise ValueError("Number of vertices not correct")
    if type(Filtr) is not Geometry.Filtration:
        raise ValueError("Not a filtration")
    if not Filtr.edgesGiven():
        raise ValueError("Filtration must be given an EdgeList")

    Ms = []
    edList = np.array(Filtr.get_edgeList())
    for eps,B in Filtr:
        M = matrix_Basis(B , NV , edList)
        Ms.append(M)

    return Ms
