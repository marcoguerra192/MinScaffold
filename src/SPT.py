# -*- coding: utf-8 -*-
# TESI: CLASSE per il problema SHORTEST PATH TREE

import numpy as np
from sympy import Matrix
from math import floor, ceil
import scipy.sparse
from scipy.sparse import csgraph
import functools
import Geometry
import multiprocessing as mlt
from multiprocessing import Pipe
import os


class SimplexGraph:
    """ Classe che implementa un grafo come matrice di adiacenza pesata, e fornisce metodi per ottenere lo shortest path tree
        a partire da un nodo arbitrario, più l'elenco dei non-tree edges.
        Input costruttore: Il numero di vertici, o il numero di vertici e la matrice dei pesi
    """

    def __init__(self, numVertices, weights, maximal=None, connection=None, eps=None):
        self.NVert = numVertices
        self.Weights = scipy.sparse.coo_matrix(weights)
        # coo_matrix può essere istanziata a partire da una matrice densa di tipo np.ndarray
        self.Weights = self.Weights.tocsr()
        # al momento ci serve che sia csr per accedere ai pesi

        self.dist = None
        self.pred = None
        self.Followers = []
        self.Sentinel = None
        self.CCC = None
        self.CCV = None
        self.Origins = None
        self.Ann = None
        self.dimH1 = None
        self.dimB1 = None
        self.dimZ1 = None
        self.Labels = -1 * np.ones((self.NVert, self.NVert), dtype=int)
        self.Support = []  # vettori di supporto
        self.SHB = None  # base minima di omologia
        self.NTE = None  # array di liste di NTE
        self.closedCycles = {}  # dizionario di cicli già chiusi
        # la chiave è la terna (source, v1, v2)
        self.allDraws = True
        self.cycleBasis = []

        # multiprocessing pipe object to return partial results
        self.pipe = connection
        self.eps = eps

        # matrici per il calcolo con la versione ricorsiva
        self.L = None  # matrice dei label m(S_i , a(e_j))
        self.C = None  # matrice m(S_i, C_j)
        self.B = None  # matrice m(S_i, C_j)
        self.S = None  # matrice S_i

        # diversifichiamo il caso in cui abbiamo la filtrazione dal caso
        # singolo step
        if (maximal == None):
            rw, cl, _ = scipy.sparse.find(self.Weights)  # righe e colonne sparse
            self.Edges = list(zip(list(rw), list(cl)))  # lista di coppie di vertici
            # scelgo solo quelli da un vertice più basso ad uno più alto
            self.Edges = [x for x in self.Edges if x[0] < x[1]]
            # funziona perchè non ci sono self loops

            def orderEdges(edgeList):
                """ Funzione che ordina la lista di edge secondo qualche criterio. Per ora il criterio è prima rispetto al primo vertice,
                poi rispetto al secondo.
                """
                return sorted(edgeList, key=lambda t: (t[0], t[1]))
            # ora è una lista ordinata di edges, senza ripetizioni!
            self.Edges = orderEdges(self.Edges)
            # può essere usata per tenere traccia dell'ordine degli edges
            self.NEdges = len(self.Edges)  # numero di edges IN SENSO INDIRETTO
            self.WEdges = []  # lista delle lunghezze (pesi) degli edges
            for e in self.Edges:
                self.WEdges.append(self.Weights[e[0], e[1]])

            self.WEdges = np.array(self.WEdges)  # trasformo in array
            self.NonZ = int(self.Weights.count_nonzero() / 2)
        else:  # se ho definito la matrice più fine della filtrazione
            # maximal è una tupla = (edgelist, weightlist)
            # la edgelist deve già essere ordinata
            self.Edges = maximal[0]
            self.NEdges = len(self.Edges)
            self.WEdges = maximal[1]
            self.NonZ = int(self.Weights.count_nonzero() / 2)

    def getEdgeList(Weights):
        """ Metodo Statico per ottenere l'edgelist da una matrice massimale
            In input solo la matrice dei pesi
        """
        rw, cl, _ = scipy.sparse.find(Weights)  # righe e colonne sparse
        edgelist = list(zip(list(rw), list(cl)))  # lista di coppie di vertici
        # scelgo solo quelli da un vertice più basso ad uno più alto
        edgelist = [x for x in edgelist if x[0] < x[1]]
        # funziona perchè non ci sono self loops

        def orderEdges(edgeList):
            """ Funzione che ordina la lista di edge secondo l'ordine lessicografico
            """
            return sorted(edgeList, key=lambda t: (t[0], t[1]))
        edgelist = orderEdges(edgelist)  # ora è una lista ordinata di edges, senza ripetizioni!
        WEdges = []  # lista delle lunghezze (pesi) degli edges
        for e in edgelist:
            WEdges.append(Weights[e[0], e[1]])
        WEdges = np.array(WEdges)  # trasformo in array

        return (edgelist, WEdges)

    def getEdgeVector(self):  # ordinamento degli archi
        return self.Edges

    def getHeatMap(self, edgeList):
        """ Prende in input una lista di coppie di vertici (v1, v2), li ordina in modo che v1 < v2, e restituisce un vettore 0-1 lungo             quanto la dimensione dello spazio vettoriale delle 1-catene, che descrive la 1-catena corrispondente.
            Se l'input contiene un edge inesistente o un self-loop, solleva ValueError
        """
        edgeList = [x if (x[0] < x[1]) else tuple(reversed(x)) for x in edgeList]  # impone v1 <= v2
        # e se v1 == v2 ?
        if (any([x[0] == x[1] for x in edgeList])):
            raise ValueError('La lista contiene un self-loop!')  # eccezione

        edgeList = list(set(edgeList))  # elimina i duplicati

        heatMap = np.zeros(self.NEdges, dtype=int)  # vettore di zeri

        for val in edgeList:
            try:
                i = self.Edges.index(val)
            except:
                errstr = "L'edge ", val, " non esiste!"
                raise ValueError(errstr)
            heatMap[i] = 1
        return heatMap

    def set_weights_by_coo(self, data, rows, col):
        """ Istanzia la matrice dei pesi nel formato sparso attraverso tre array data, rows e col"""
        self.Weights = scipy.sparse.coo_matrix((data, (rows, col)), shape=(self.NVert, self.NVert))

    def computeShortestPath(self, withDistances=False):
        """ Restituisce la matrice dei predecessori nello Shortest Path Tree. Ogni riga si riferisce ad un nodo
            di origine dello SPT. Se l'argomento opzionale è vero ritorna anche la matrice delle distanze
        """
        # va convertito in un formato su cui può fare i calcoli
        (self.dist, self.pred) = csgraph.shortest_path(
            self.Weights.tocsr(), directed=False, return_predecessors=True)

        # Aggiungiamo il calcolo dei successori. Utile da fare una volta invece di n
        # Viene utile una struttura di lista^3
        for source in range(self.NVert):  # per ogni sorgente
            link = self.pred[source, :]  # predecessori
            forward = [[] for i in range(self.NVert)]  # lista di liste vuote lunga NVert
            for (foll, prev) in enumerate(link):
                if (prev != -9999):  # se è connesso e non è source
                    forward[prev].append(foll)
            self.Followers.append(forward)
        # non c'è nessun bisogno di restituire una matrice densa
        # if withDistances:
        #     return (self.pred, self.dist)
        # else:
        #     return self.pred

    def shortestPathTree(self, source, withD=False, justEdges=False):
        """ Restituisce la matrice dello shortest path tree centrata nel vertice source. Se withD è vero
            la matrice è pesata, altrimenti è solo 0-1. Se source non è un vertice valido solleva un'eccezione
            I VERTICI VANNO DA 0 A N-1!!
            Se justEdges == True restituisce solo i non tree edges
            Input: L'indice del nodo sorgente, opzionale un bool per le distanze
            Output: La matrice sparsa del grafo SPT
        """
        if (type(source) is not int or source < 0 or source >= self.NVert):  # da qui in poi source è un valore affidabile
            raise Exception('Indice del vertice non valido!')

        if withD:
            if (self.dist is None or self.pred is None):
                _ = self.computeShortestPath(withD)
        else:
            if self.pred is None:
                _ = self.computeShortestPath(withD)

        # definiamo una matrice sparsa della giusta dimensione che andrà in output
        # SPT = scipy.sparse.coo_matrix(self.NVert,self.NVert) # NO, SPRECA SOLO TEMPO!
        # estraiamo dalla matrice dei precedenti la riga che ci serve
        link = np.array(self.pred[source, :])
        # predisponiamo i vettori data, rows e col per creare la matrice del SPT
        data = []
        rows = []
        col = []  # man mano faremo append dei dati in queste liste
        # scorriamo il vettore e creiamo i link
        for i in range(len(link)):
            prev = link[i]
            if (prev == -9999):  # non appartiene alla stessa componente connessa => non ha predecessori
                pass
            else:
                rows.append(prev)
                col.append(i)  # aggiungi un edge dal predecessore del nodo i-esimo al nodo i-esimo
                rows.append(i)  # devo aggiungere anche il simmetrico!
                col.append(prev)
                if withD:  # se devo aggiungere le distanze le cerco in self.dist
                    data.append(self.dist[prev, i])
                    # aggiungo anche il simmetrico (stessa distanza)
                    data.append(self.dist[prev, i])
                else:  # se no inserisco solo 1 per indicare l'edge fra prev e i
                    data.append(1)
                    data.append(1)

        if withD:
            data = np.array(data, dtype=float)
        else:
            data = np.array(data, dtype=int)
        rows = np.array(rows)
        col = np.array(col)
        SPT = scipy.sparse.coo_matrix((data, (rows, col)), shape=(self.NVert, self.NVert))
        # crea matrice del grafo
        # Per trovare i non-tree edge: trovare gli elementi di Weights che NON SONO in SPT
        # restituisce vettori di righe, colonne e valori non nulli
        rw, cl, _ = scipy.sparse.find(self.Weights)
        # ignoro il vettore di dati, che non ci interessa
        edgeSPT = list(zip(list(rows), list(col)))  # creo una lista di edge per SPT
        edgeW = list(zip(list(rw), list(cl)))  # stessa cosa per il grafo originale
        # MA CONTIENE ANCHE TUTTI GLI EDGES NELLE COMPONENTI NON CONNESSE!
        # Affinchè il metodo funzioni, è necessario che il nodo source non abbia -9999 per distinguerlo da quelli davvero disconnessi
        link[source] = source  # creiamo un "finto" self-loop
        edgeW = [x for x in edgeW if link[x[0]] != -9999]
        # limito a solo gli edge fra nodi che sono raggiungibili da source
        NTE = list(set(edgeW) - set(edgeSPT))  # differenza insiemistica tra i set di edges
        # contiene ancora entrambe le versioni dell'edge (v1,v2 e v2,v1). Imponiamo solo quella (v1,v2) con v1<v2
        NTE = [x for x in NTE if (x[0] < x[1])]

        if justEdges:
            return NTE
        else:
            return (SPT, NTE)

    def spanningTree(self):
        """ Calcola uno spanning tree a partire dalla stessa matrice dei predecesori di SPT. Questa volta, oltre a lavorare su tutte le               componenti connesse, restituisce tutti i sentinel edges.
        """
        if self.pred is None:
            _ = self.computeShortestPath(True)
        # prepariamo le liste per la matrice del grafo
        rows = []
        col = []
        data = []
        self.CCV = list(np.zeros(self.NVert, dtype=int))  # lista indice delle componenti connesse
        self.CCC = 0  # contatore delle componenti connesse
        self.Origins = []  # lista lunga come il numero di CC. In posizione k contiene l'origine della componente connessa k+1-esima
        # che sarà necessario utilizzare per calcolare i sentinel edges!
        while (0 in self.CCV):  # finchè non ho assegnato ogni vertice ad una componente
            self.CCC += 1
            vert = self.CCV.index(0)  # trova il primo vertice non assegnato ad alcuna componente
            # Questo vertice diventerà l'origine della CC!! Sarà necessario usare lui per calcolare i sentinel edges!
            self.Origins.append(vert)  # questa CC origina da vert
            self.CCV[vert] = self.CCC  # assegna questo vertice alla nuova componente
            link = np.array(self.pred[vert, :])  # estrai i predecessori connessi a vert
            for i in range(len(link)):
                prev = link[i]
                if (prev == -9999):  # se è disconnesso fai nulla
                    pass
                else:
                    self.CCV[i] = self.CCC  # se è connesso assegna lo stesso indice di componente
                    rows.append(prev)
                    # aggiungi un edge dal predecessore del nodo i-esimo al nodo i-esimo
                    col.append(i)
                    rows.append(i)  # devo aggiungere anche il simmetrico!
                    col.append(prev)
                    data.append(1)
                    data.append(1)
        data = np.array(data, dtype=int)
        rows = np.array(rows)
        col = np.array(col)
        ST = scipy.sparse.coo_matrix((data, (rows, col)), shape=(
            self.NVert, self.NVert))    # matrice del grafo Spanning Tree
        # per trovare i sentinel edges. Questo spanning tree è già generato a partire dall'origine di ogni CC!
        rw, cl, _ = scipy.sparse.find(self.Weights)  # tutti gli edge del grafo originale
        edgeW = list(zip(list(rw), list(cl)))  # lista edge grafo originale
        edgeST = list(zip(list(rows), list(col)))  # lista edge spanning tree
        self.Sentinel = list(set(edgeW) - set(edgeST))  # differenza insiemistica
        # contiene ancora entrambe le versioni dell'edge (v1,v2 e v2,v1). Imponiamo solo quella (v1,v2) con v1<v2
        self.Sentinel = [x for x in self.Sentinel if (x[0] < x[1])]
        self.Sentinel = sorted(self.Sentinel, key=lambda x: (x[0], x[1]))
        return (ST, self.Sentinel, self.CCV)

    def closeCycle(self, E, force=None):
        """
        Funzione che prende in input un sentinel edge E, e sfruttando la matrice
        self.pred e l'ordinamento self.Edges, restituisce un vettore dello spazio
        delle 1-catene che descrive il ciclo identificato da E nello Shortest Path Tree.
        Lo SPT è quello radicato nel nodo ORIGINE DELLA CC DI E!
        Se l'input non è quello atteso, solleva ValueError.
        SE E NON È UN SENTINEL EDGE NON SO BENE COSA SUCCEDA!
        Se è definito force, forza il nodo sorgente ad essere quello. Utile per
        generare il candidate set.
        Sfrutta la struttura self.closedCycles per verificare se una certa chiusura
        è già stata calcolata
        """

        if (force is not None):  # se è fissata l'origine
            key = (force, E[0], E[1])  # la chiave è (source, v1,v2)
            try:  # trova se è già stato calcolato
                cycle = self.closedCycles[key]
                return cycle
            except:
                pass

        try:
            # trova l'heatmap del sentinel edge. Gestisce l'errore se E non è giusto
            cycle = self.getHeatMap([E])
        except Exception as ex:
            raise ex
        # ora cycle contiene l'heatmap del solo sentinel edge.
        # Proviamo a fare un controllino che sia un sentinel edge. Se self.Sentinel non esiste può comunque far danni
        if (self.Sentinel is not None and force is None):  # solo se non è definito force
                                                        # (in quel caso ci sta che non siano sentinel edges)
            if (E not in self.Sentinel and tuple(reversed(E)) not in self.Sentinel):
                raise ValueError("E non è un sentinel edge!")

        # fissiamo i bordi del sentinel edge
        v1 = E[0]
        v2 = E[1]

        if (force is not None):
            if (force < 0 or force >= self.NVert):
                raise ValueError("Parametro force non valido!")
            else:
                source = force
        else:
            # troviamo il NODO ORIGINE della CC di E
            # -1 perchè le CC partono da 1 ma i vettori da 0. Sarebbe uguale usare v2
            source = self.Origins[self.CCV[v1] - 1]
            key = (source, E[0], E[1])  # calcoliamo la chiave del ciclo da chiudere
        # Calcoliamo i pred, se non è già stato fatto
        if (self.pred is None):
            self.pred = self.computeShortestPath(False)

        link = self.pred[source, :]  # vettore dei precedenti rispetto a source
        pr = link[v2]
        if (pr == -9999 and v2 != source):  # significa che v2 è irraggiungibile da source, ci deve essere stato un errore
            raise ValueError(
                'Critical Error! Source e v2 dovrebbero essere per costruzione nella stessa CC')
        follow = v2
        while (pr != -9999):  # segue all'indietro i precedenti verso source (che ha link -9999)
            cycle += self.getHeatMap([(pr, follow)])
            follow = pr
            pr = link[pr]

        # Ora stessa cosa per v1, ma gli edge in comune fra i due path vanno eliminati => SOMMA % 2

        pr = link[v1]
        if (pr == -9999 and source != v1):  # significa che v1 è irraggiungibile da source, sono in due componenti connesse diverse
            raise ValueError(
                'Critical Error! Source e v1 dovrebbero essere per costruzione nella stessa CC')
        follow = v1
        while (pr != -9999):  # finchè non torno a source
            cycle += self.getHeatMap([(pr, follow)])
            follow = pr
            pr = link[pr]
        # ora alcuni cicli saranno stati contati 2 volte. Quelli in comune vanno eliminati!
        cycle = cycle % 2

        # ora cycle contiene la descrizione vettoriale del ciclo identificato dal
        # sentinel edge E, rispetto all'ordinamento self.Edges

        # aggiorniamo il dizionario dei cicli chiusi
        self.closedCycles[key] = cycle

        return cycle

    def getCycleBase(self):
        """ Calcola una base dei cicli a partire dai sentinel edges. Per ottenere l'annotazione
        """
        # se ancora non l'ha fatto calcola i sentinel edges
        if (self.Sentinel is None):
            _ = self.spanningTree()

        for e in self.Sentinel:
            self.cycleBasis.append(self.closeCycle(e))

    def getAnnotation(self, d2):
        """
        d2: border matrix of the set of faces of the simplicial complex, borders are written as rows.
        Z: basis of 1-dim-cycles. Each row is a cycle obtained from a spanning tree and a sentinel edge.
        returns an annotation of sentinel edges. Non sentinel edges have annotation equal to 0.
        """
        Z = self.cycleBasis

        def low(col):
            """
        col: 1-dimensional array.
            gets the index of the "lowest" element in col different from 0.
            if col=0 then low = -1
            """
            l = -1
            for i in range(len(col)):
                if col[i] > 0:
                    l = i
            return l

        lowSet = {}  # dictionary with low indexes and relative rows

        if len(d2) == 0:  # controlla se d2 è vuoto!
            dimB1 = 0
            i = 0
        else:
            i = 0
            while i != len(d2):
                lowRowi = low(d2[i])
                while lowRowi in lowSet.keys():
                    d2[i] = (d2[i]+d2[lowSet[lowRowi]]) % 2
                    lowRowi = low(d2[i])
                if lowRowi > -1:
                    lowSet[lowRowi] = i
                    i = i+1
                else:
                    d2 = np.delete(d2, (i), axis=0)
            dimB1 = i  # dimensione dello spazio dei bordi

        if (dimB1 != 0):
            Z = np.concatenate((d2, Z), axis=0)
        else:
            pass  # lasciamo Z
        # we start the reduction from row dimB1
        totRow = len(Z)
        reductionMatrix = np.identity(totRow, dtype=int)
        Id = np.identity(totRow, dtype=int)
        elementsToDelete = []

        while i != totRow:
            lowRowi = low(Z[i])
            while lowRowi in lowSet.keys():
                Z[i] = (Z[i]+Z[lowSet[lowRowi]]) % 2
                reductionMatrix[i] = (reductionMatrix[i]+Id[lowSet[lowRowi]]) % 2
                lowRowi = low(Z[i])

            if lowRowi > -1:
                lowSet[lowRowi] = i
                i = i+1
            else:
                elementsToDelete.append(i)
                i = i+1

        # eliminate coordinates of cycles that are borders:
        reductionMatrix = np.delete(reductionMatrix, elementsToDelete, axis=1)
        reductionMatrix = np.delete(reductionMatrix, range(dimB1), axis=1)
        A = np.delete(reductionMatrix, range(dimB1), axis=0)
        """observation: the number of rows of A is the dimension of the 1-dim-cycle group;
           the number of columns is the dimension of the 1st homology group
        """
        self.dimB1 = dimB1
        self.dimZ1 = np.shape(A)[0]
        self.dimH1 = np.shape(A)[1]
        self.Ann = np.matrix(A).transpose()

        # OTTIMIZZAZIONE!
        # Qui verrebbe bene creare un dizionario di coppie edge-annotazione
        # per velocizzare la ricerca negli step successivi. Edge nel formato
        # (v1, v2), con v1<v2
        self.AnnDict = {}
        for i, e in enumerate(self.Sentinel):
            ann = self.Ann[:, i]
            self.AnnDict[e] = ann

        # MULTIPROCESSING! If pipe exists send message
        if self.pipe is not None:
            self.pipe.send([self.eps, mlt.current_process().pid, 0, self.dimH1, self.NonZ])

        return (self.Ann, self.dimB1, self.dimZ1, self.dimH1)

    def innerProd(self, S, C):
        """ Scalar Product over Z2 of support vector S and the ANNOTATION of cycle C
        """
        if (len(S) != len(C)):
            raise ValueError("Dimensioni non compatibili!")

        return np.dot(np.array(S).transpose(), np.array(C)) % 2

    def computeLabels(self, Sup):
        """ Calcola i labels rispetto al support vector S_i Sup
        """
        if (self.Ann is None):
            raise ValueError("Non è stata calcolata l'annotazione degli edges!")
        for p in range(self.NVert):  # per ogni root
            #link = self.pred[p,:]
            forward = self.Followers[p]
            """ Utilizziamo un metodo tipo push pop: ad ogni nodo aggiungiamo la
            lista dei successori in coda, poi pop di uno e avanti così """
            self.Labels[p, p] = 0  # il label di source è 0
            driver = []  # lista di push/pop DI TUPLE (PREVIOUS,FOLLOWER)
            # iniziamo dai sucessori di source
            add = [(p, x) for x in forward[p]]
            driver.extend(add)
            while (len(driver) != 0):  # finchè non ho esaurito i successori
                (prev, foll) = driver.pop(0)  # pop del primo elemento (BREADTH FIRST!)
                lab = self.Labels[p, prev]  # leggiamo il label del nodo precedente
                edge = (prev, foll) if prev < foll else (foll, prev)  # scriviamo l'edge
                try:  # leggiamo l'annotazione dell'ege
                    ann = self.AnnDict[edge]
                except KeyError:
                    self.Labels[p, foll] = lab  # non è un sentinel e quindi non cambia
                else:
                    # calcola il prodotto scalare in Z2 e somma
                    self.Labels[p, foll] = (lab + self.innerProd(Sup, ann)) % 2
                # Ora aggiungiamo alla lista driver i successori di foll e avanti
                add = [(foll, x) for x in forward[foll]]
                driver.extend(add)

        # return self.Labels # inutile restituire una struttura pesante

    def fixNTE(self):
        """ Calcola un array di NTE per ogni sorgente p, e la salva in self.NTE
        """
        self.NTE = []
        for p in range(self.NVert):
            nte = self.shortestPathTree(p, justEdges=True)
            for e in nte:
                self.NTE.append((p, e))
        # ora tutti i NTE sono calcolati una volta per tutte

    def resetLabels(self):
        """ Resetta i labels per poterli calcolare rispetto al nuovo vettore di supporto
        """
        self.Labels = -1 * np.ones((self.NVert, self.NVert), dtype=int)

    def cycleAnnotation(self, C):
        """ Calcola l'annotazione del ciclo C (somma delle ann degli edges)
        """
        # seleziona gli edges del ciclo
        C = map(lambda x: bool(x), C)  # rendilo bool
        edges = [e for (e, filt) in zip(self.Edges, C) if filt]
        ann = np.zeros((self.dimH1, 1), dtype=int)
        for e in edges:
            try:
                a = self.AnnDict[e]
                ann = (ann + a) % 2
            except KeyError:
                pass  # se non è un sentinel fai nulla

        return ann

    def findShortestNonOrtho(self, i):
        """ Funzione che genera una lista di cicli non-ortogonali (m = 1) a Sup
            che è l' i-esimo vettore di supporto.
            I NTE seguono la convenzione lessicografica
            If allDraws ritorna una lista di tutti i cicli minimi a parità!
        """
        # bisogna controllare che sia stato generato tutto il resto
        # calcolare i labels
        # self.resetLabels() # riporta i labels a -1 Con i followers non serve più

        Sup = self.S[:, i-1]
        _ = self.computeLabels(Sup)  # è da fare lo stesso

        def checkOrt(s, e):
            """ In input una sorgente dello SPT e un edge scritto giusto
            """
            try:
                ann = self.AnnDict[e]  # trova l'annotazione dal dizionario
            except KeyError:  # se non appartiene ai sentinel vale 0
                lab = 0
            else:
                lab = self.innerProd(Sup, ann)  # calcola il prodotto

            return (self.Labels[s, e[0]] + self.Labels[s, e[1]] + lab) % 2
            # return (self.L[e[0]] + self.L[s,e[1]] + lab ) %2

        candidates = []  # lista di tuple (origine, NTE)

        # selezioniamo solo quelli non ortogonali
        candidates = [x for x in self.NTE if checkOrt(x[0], x[1]) == 1]

        # generiamo i vettori dei cicli a partire da (source, non-tree edge)
        candidates = [self.closeCycle(x[1], force=x[0]) for x in candidates]

        # definiamo un metodo per calcolare la lunghezza di un ciclo
        def lenCycle(C):
            """ Calcola la lunghezza di un ciclo come prodotto scalare tra il
                vettore dei pesi self.WEdges e la descrizione del ciclo nella
                base self.Edges
            """
            return np.dot(self.WEdges, C)

        # list of tuples (ciclo, lunghezza)
        candidates = list(zip(candidates, map(lenCycle, candidates)))

        # MULTIPROCESSING!
        if self.pipe is not None:
            self.pipe.send([self.eps, mlt.current_process().pid, i, self.dimH1, self.NonZ])

        if self.allDraws:  # restituisce tutti i pareggi
            shortest = candidates[0][1]
            minList = []
            for x in candidates:
                if x[1] < shortest:
                    minList = [x]
                    shortest = x[1]
                elif x[1] == shortest:
                    minList.append(x)
                else:
                    pass

            target_annotation = self.cycleAnnotation(
                minList[0][0])  # annotation of the chosen cycle
            minList2 = [minList[0][0]]
            del minList[0]  # remove the chosen cycle from minList
            for c in minList:
                cAnn = self.cycleAnnotation(c[0])
                if not np.count_nonzero(target_annotation - cAnn):
                    counter = 0
                    for minC in minList2:
                        if np.count_nonzero(c[0] - minC):
                            counter += 1
                    if counter == len(minList2):
                        minList2.append(c[0])

            # res = [ c[0] for c in minList] # only want the edges, not lengths
            # res = [ np.array( c , dtype=int ) for c in res ] # make it a np.array
            res = [np.array(c, dtype=int) for c in minList2]
            return res

        else:  # trovane solo uno
            def min_len(x, y): return x if x[1] <= y[1] else y
            minCycle = functools.reduce(min_len, candidates)
            minCycle = minCycle[0]  # don't care about lengths, just edges
            minCycle = np.array(minCycle, dtype=int)

            return minCycle

    def updateSup(self, newC, index):
        """ Funzione che aggiorna i support vector per ogni nuovo ciclo minimo
            della base di omologia. Fa in pratica un Grahm-Schmidt in Z2
            Deve calcolare l'annotazione del ciclo.
            index deve andare da 0 a g-1
        """

        # self.SHB.append( newC ) # aggiungi il nuovo ciclo # QUESTO CAMBIA PER LA
        # VERSIONE RICORSIVA
        sup_i = self.Support[index]  # support vector del nuovo ciclo
        AnnNewC = self.cycleAnnotation(newC)  # annotazione del nuovo ciclo

        for j in range(index+1, self.dimH1):  # per S_j da i+1 a g
            sup_j = self.Support[j]
            self.Support[j] = (sup_j + sup_i * self.innerProd(AnnNewC, sup_j)
                               ) % 2  # somma ( =sottrae) la proiezione

    def initSup(self, Draws=True):
        """ Function to initialize the Support vectors structures self.Support and
            self.S. Plus it preallocates L,C and B. Then fixes the non-tree edges.
            Also, initializes the Gemoetry.Basis for self.SHB (Edges are computed)
            REQUIRES THE ANNOTATION TO HAVE BEEN COMPUTED!
        """
        self.S = np.eye(self.dimH1, dtype=np.int64)
        for i in range(self.dimH1):
            self.Support.append(self.S[:, i])

        # init the matrices
        self.L = np.zeros((self.dimH1, len(self.Sentinel)), dtype=np.int64)
        self.C = np.zeros((len(self.Sentinel), self.dimH1), dtype=np.int64)
        self.B = np.zeros((self.dimH1, self.dimH1), dtype=np.int64)

        # compute labels as a starter
        for i in range(self.dimH1):
            for j in range(len(self.Sentinel)):
                ann = self.AnnDict[self.Sentinel[j]]
                self.L[i, j] = self.innerProd(self.S[:, i], ann)

        # fix non-tree edges as we need them for the base case
        _ = self.fixNTE()

        # FIND A FILTER BETWEEN EDGES AND SENTINEL EDGES
        Filter = np.array([1 if (x in self.Sentinel) else 0 for x in self.Edges])
        self.SentFilter = np.nonzero(Filter)

        # Let us initialize the variable self.SHB as a class Basis object of the
        # Geometry module

        self.SHB = Geometry.Basis(self.NEdges, edgelist=self.Edges)

        self.allDraws = Draws

    # def  Update(self,i,k):
    #     """ Recursive version of the UpdateSup function.
    #         FOR NOW: It works nothing like the paper. It simply pops the last
    #         basis vector that was added to SHB, finds the corresponding Support
    #         vector, and updates the old way via Grahm-Schmidt
    #     """
    #     newCycle = self.SHB[-1] # è una tupla con anche la lunghezza!
    #     newCycle = newCycle[0] # ora è solo l'array
    #     ind = len(self.SHB)-1
    #     self.updateSup(newCycle,ind)

    def mulz2(self, a, b):
        return np.mod(np.matmul(a, b), 2).astype(np.float64)

    def sumz2(self, a, b):
        return np.mod(a+b, 2).astype(np.float64)

    def inv_z2(self, a):
        # Input/Output are numpy arrays
        # print(a)
        det = np.linalg.det(a)

        return np.mod(np.round(np.linalg.inv(a)*det), 2).astype(np.int64)
        return np.array(Matrix(a).inv_mod(2)).astype(np.int64)

    def UpdateMehlhorn(self, i, k):
        """ Update function optimized for the divide et impera recursive approach
            as per Mehlhorn (2004), Dey (2018).
            Uses instance variables L,C,B and S. Receives recursion parameters
            i and k.
            Improves (?) from O(m^3) to O(gn^2) for low-connected graphs and to
            O(gn^2 log n) for highly-connected ones.
        """
        # MULTIPROCESSING!
        if self.pipe is not None:
            self.pipe.send([self.eps, mlt.current_process().pid,
                            self.SHB.Card, self.dimH1, self.NonZ])
        # define short for recursion indices
        lam = i + floor(k/2)-1
        mu = i+k-1

        # slice matrix B to obtain X and Y
        X = self.B[i - 1:lam, i-1:lam]
        Y = self.B[lam+1-1:mu, i-1:lam]

        # By definition of A
        A = self.mulz2(Y, self.inv_z2(X))

        # MULTIPROCESSING!
        if self.pipe is not None:
            self.pipe.send([self.eps, mlt.current_process().pid,
                            self.SHB.Card, self.dimH1, self.NonZ])

        # Update Support vectors
        S1 = np.zeros((self.dimH1, mu - lam + 1), dtype=int)
        S1 = self.sumz2(self.S[:, lam+1-1:mu], self.mulz2(self.S[:, i-1:lam], A.T))
        self.S[:, lam+1-1:mu] = S1

        # Mirror in the self.Sup structure
        for col in range(lam, mu):
            self.Support[col] = self.S[:, col]

        # Update Labels
        # Generate W and Z
        m, _ = A.shape
        W = np.concatenate((A, np.eye(m)), axis=1)

        # MULTIPROCESSING!
        if self.pipe is not None:
            self.pipe.send([self.eps, mlt.current_process().pid,
                            self.SHB.Card, self.dimH1, self.NonZ])

        Z = self.L[i-1:mu, :]
        # New labels are
        #L1 = np.zeros( ( mu - lam + 1 , len(self.Sentinel) ) , dtype=int )

        L1 = self.mulz2(W, Z)
        self.L[lam+1-1:mu, :] = L1

        # At last, update matrix B, which has changed since L has changed and
        # potentially C has changed too

        self.B = self.mulz2(self.L, self.C)

        # MULTIPROCESSING!
        if self.pipe is not None:
            self.pipe.send([self.eps, mlt.current_process().pid,
                            self.SHB.Card, self.dimH1, self.NonZ])

    def projOnSent(self, C):
        """
            Compute the Sentinel-Edge trace of cycle C. Return a vector that is
            as long as self.Sentinel.
            Uses the structure self.SentFilter which projects a cycle onto the
            space of sentinel edges.
            In case it is a list of draws, pick the first!
        """
        if type(C) is list:  # in case it is a list of draws, pick the first!
            C = C[0]

        C = np.array(C, dtype=int)
        # print(C.shape)
        return C[self.SentFilter]

    def ExtendBasis(self, i, k):
        """ Recursive procedure to compute the minimal homology basis. Follows
            Mehlhorn(2004) and Dey (2018).
            Will require using the matrices to compute labels, to update sup andrà
            all.
        """
        if k == 0:  # If homology is trivial it returns the empty basis
            return
        if k == 1:  # BASE CASE. Will search for the shortest non-orthogonal cycle
            #print("Base case, i = ",i)
            newCycle = self.findShortestNonOrtho(i)

            # save the new cycle to the SHB
            self.SHB.addEl(newCycle)
            # compute the Sentinel-edge trace of the new cycle
            self.C[:, i-1] = self.projOnSent(newCycle)
            self.B[:, i-1] = self.mulz2(self.L, self.C[:, i-1])
        else:
            self.ExtendBasis(i, floor(k/2.0))  # extend by floor(k/2) elements
            self.UpdateMehlhorn(i, k)  # updates support vectors
            self.ExtendBasis(i + floor(k/2.0), ceil(k/2.0))

    # THE END
