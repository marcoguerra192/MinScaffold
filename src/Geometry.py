#GEOMETRY

import numpy as np
from numpy import linalg as la
import networkx as nx
import matplotlib.pyplot as plt

class Filtration:
    """
    Define a Filtration object. Will be a sequence indexed by the epsilon thresholds
    """

    def __init__(self, edL = None):

        self.Card = 0 # cardinality of the filtration
        self.Elements = []
        self.edgeList = edL
        self.index=0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.Card:
            raise StopIteration
        else:
            el = self.Elements[self.index]
            self.index += 1
            return (el[0] , el[1])



    def add(self , eps , Obj):
        """
        """
        if self.Card == 0:
            self.Elements.append( (eps , Obj) )
            self.Card = 1

        else:
            if type(Obj) == type( self.Elements[0][1] ):
                self.Elements.append( (eps , Obj) )
                self.Card += 1
            else:
                raise ValueError("Types do not match")

    def sort_by_eps(self):
        """
        """
        self.Elements = sorted(self.Elements , key =lambda el: el[0] )

    def isEmpty(self):
        return (self.Card == 0)

    def as_list(self):
        return self.Elements

    def length(self):
        return self.Card

    def size(self):
        if self.edgeList is not None:
            return len(self.edgeList)
        else:
            return None
    def get_edgeList(self):
        return np.array(self.edgeList)
    def set_edgeList(self, edL):
        # We must include some checks here!
        self.edgeList = edL

    def edgesGiven(self):
        return not self.edgeList is None

class Basis:
    """
        Implements a basis of cycles. Needs to know the EdgeList.
        Draws must be transparent.
    """
    def __init__(self, Len, edgelist=None):

        self.length = Len
        self.EdgeList = edgelist
        self.Card = 0

        self.Elements = []

    def isEmpty(self): # check if the basis has no elements
        return (self.Card == 0)

    def card(self):
        return self.Card

    def size(self):
        return self.length

    def addEl(self, cycle):
        if type(cycle) is np.ndarray: # è un ciclo unico
            if cycle.shape != (self.length,): # non sono sicuro che funzioni
                raise ValueError("Format is not correct")
            if len(cycle) != self.length:
                raise ValueError("Length of cycle is not correct")
            #if cycle.dtype is not int :    ## This doesn't seem to work! Says dtype is dtype(int64)
            #    raise ValueError("The cycle is not well-expressed")

            self.Card += 1
            newEl = ( 1 , [cycle] )
            self.Elements.append(newEl)

        elif type(cycle) == list: # è una lista di pareggi
            L = len(cycle)
            cy = []
            if L == 0:
                raise ValueError("Not a cycle")
            for el in cycle:
                if type(el) is not np.ndarray:
                    raise ValueError("Not a list of cycles")
                #if el.dtype is not int:   ## This doesn't seem to work! Says dtype is dtype(int64)
                #    raise ValueError("Not a list of cycles")
                if len(el) != self.length:
                    raise ValueError("Length of cycle is not correct")
                cy.append(el)

            self.Card = self.Card + 1
            newEl = ( L , cy )
            self.Elements.append(newEl)
        else:
            raise ValueError("Not the correct formatting af a cycle")

    def profile(self):
        return  [ x[0] for x in self.Elements  ]

    def as_list(self):
        """

        """
        res = []
        for el in self.Elements:
            res.append(el[1])
        return res

    def as_plain(self):
        res = []
        for el in self.Elements:
            res.extend(el[1])
        return res

    def smear(self):
        """
            Return a list of np.arrays with fractional values, incidences smeared
            over the edges.
        """
        res = []
        for el in self.Elements:
            newC = np.zeros( (self.length,) , dtype=float )
            for c in el[1]:
                newC = np.add(newC, c)
            newC = newC / float(el[0])
            res.append(newC)
        return res

def genMatrix(NV, Quant):
    """Create a symmetric random matrix with zero diagonal. Filter it according to Quant"""
    W = np.matrix(np.random.rand(NV,NV))
    W = W + W.transpose()
    for i in range(NV):
        W[i,i] = 0
    Filter = Quant * np.max(W)
    W[ W<Filter] = 0
    return W


def genFullEpsList(W):
    """
        Input: a weighted adjacency matrix, of suitable numpy type
        Outputs: a list of thresholds that induce the full filtration
    """
    W = np.array(W, dtype=float) # unique works on numpy array only

    # if not (W.dtype == np.float64 or W.dtype == np.float32):
    #     raise ValueError('The adj matrix is not float type')

    vals =  np.unique(W).tolist()
    vals = [ x + 0.001 for x in vals ]

    return vals

def matCompare(A,B):
    """
    Compare matrices A and B via several norms of A-B. If no matrix B is provided, norms are computed on A alone.
    It may be necessary that arguments be (weighted) adjacency matrices, i.e. symmetric with non-negative entries.
    """
    cmp = {}

#     if not (type(A) is np.ndarray and type(B) is np.ndarray):
#         raise ValueError('Invalid matrices passed')

#     if B is not None:
#         A = A-B

    m , n = A.shape

    cmp['2'] = (la.norm(A), la.norm(B))
    cmp['1'] = (la.norm(A,1) , la.norm(B))

    cmp['inf'] = ( la.norm( A.reshape(1,m*n) , np.inf) , la.norm( B.reshape(1,m*n) , np.inf))

    return cmp

def netCompare(A, B=None):
    """
    Compute the networkx comparisons, like degree distribution, betweenness, others
    INPUT: the two nx graphs
    OUTPUT: Boh!? The plottable objects?
    """
    res = {}

    res['deg'] = ( A.degree() , B.degree() )

    # degree centrality
    res['deg_cent'] = \
    (nx.algorithms.centrality.degree_centrality(A), \
    nx.algorithms.centrality.degree_centrality(B))

    # betweenness_centrality
    res['bet_cent'] = \
    (nx.algorithms.centrality.betweenness_centrality(A), \
    nx.algorithms.centrality.betweenness_centrality(B))

    # closeness centrality
    res['clsn_cent'] = \
    (nx.algorithms.centrality.closeness_centrality(A), \
    nx.algorithms.centrality.closeness_centrality(B))

    # eigenvector centrality
    res['eig_cent'] = \
    (nx.algorithms.centrality.eigenvector_centrality(A), \
    nx.algorithms.centrality.eigenvector_centrality(B))

    # clustering coefficients
    res['cluster'] = (nx.clustering(A), nx.clustering(B))

    # average clustering
    res['avg_cl'] = (nx.average_clustering(A),nx.average_clustering(B))

    return res


def compare(A,B):
    """
    """

    mat = matCompare(A,B)
    A = nx.from_numpy_matrix(A, parallel_edges=False)
    B = nx.from_numpy_matrix(B, parallel_edges=False)
    net = netCompare(A,B)

    return { **mat , **net }



## STUB GENERATORS OF RANDOM FAMILIES OF GRAPHS

def genWSFamily(n, kList, pList):
    """
    Generate a family of Watts-Strogatz graphs of given parameters
    """

    res = []

    for k in kList:
        for p in pList:
            res.append(nx.generators.random_graphs.watts_strogatz_graph(n,k,p))


    return res
