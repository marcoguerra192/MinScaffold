import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from math import sin, cos, pi
from random import random
from SPT import SimplexGraph
import scipy.sparse
import time
import datetime
import multiprocessing
from multiprocessing import Pool, Pipe
import _pickle as cPickle
import Geometry
from sys import stdout, stderr
import os


def getTriangles(A):
    '''
    returns set of triangles in a graph given its adjacency matrix
    '''
    n = len(A)
    tri = []
    for vertex in range(n):
        vList = np.nonzero(A[vertex,vertex:])
        vList = [x for x in vList[1]] #NB: qui c'era uno zero che non faceva più funzionare l'algoritmo
        vList = [i + vertex for i in vList] #list of vertices adjacent to vertex
        for i in range(len(vList)-1):
            for j in range(i+1,len(vList)):
                if A[vList[i],vList[j]] > 0:
                    tri.append([vertex,vList[i],vList[j]])
    return tri

def getD2(A, edgesList):
    '''
    returns 2-boundary matrix of the clique complex, given the adjacency matrix of the graph
    '''
    triangles = getTriangles(A)
   # print(triangles)

    d2 = []
    n = len(edgesList)
    #print(n)
    for row in triangles:
        #pdb.set_trace()
        newTriangle = [0 for i in range(n)]
        newTriangle[edgesList.index( (row[0],row[1] ) )] = 1
        newTriangle[edgesList.index((row[0],row[2]) )] = 1
        newTriangle[edgesList.index( (row[1],row[2]) )] = 1
        d2.append(newTriangle)
    d2 = np.array(d2)

    return d2


def points2adj(P, epsilon):
    '''
    given a set of points and a threshold returns the weight and adjacency matrix of the corresponding graph
    points is a list where each element is the coordinates of the point
    '''

    def dist(p1,p2):
        '''
        square of euclidean distance
        p1 and p2 need to be list of the same length
        '''
        return math.sqrt(sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))

    n = len(P)
    W=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            W[i,j]= dist(P[i],P[j])
    W = W + W.T #weight matrix
    W[W > epsilon] = 0 # meglio eliminare i valori STRETTAMENTE maggiori di epsilon

    A = (W >0).astype(int)
    return W , A

def filterMatrix(W,epsilon):
    Wb = np.matrix(W)
    Wb[Wb > epsilon] = 0 # meglio eliminare i valori STRETTAMENTE maggiori di epsilon
    return Wb

def plotCycle(ax,cic,edList,P):
    for i in range(len(cic)):
        if cic[i] > 0:
            edge = edList[i]
            p1 = edge[0]
            p2 = edge[1]
            line = lines.Line2D([P[p1][0],P[p2][0]],[P[p1][1],P[p2][1]],color='r')
            #line = lines.Line2D([P[edList[cic[i]][0]][0],P[edList[cic[i]][1]][0]],[P[edList[cic[i]][0]][1],P[edList[cic[i]][1]][1]], color='r')
            ax.add_line(line)

def sampleCircle(x0,y0,r,n, noise):
    P = []
    for i in range(n):
        theta = random()*2*pi
        R = r+noise*random()
        xp = x0 + R*cos(theta)
        yp = y0 + R*sin(theta)
        P.append([xp,yp])
    return P

def getEL(A):
    edgesList = []
    for i in range(len(A)):
        for j in range(i,len(A)):
            if A[i,j] > 0:
                edgesList.append([i,j])
    return edgesList

def plot_examples(SHBi,Wi,maximal, P):
    fig, ax = plt.subplots()

    x = [i[0] for i in P]
    y = [i[1] for i in P]
    ax.scatter(x,y)
    for row in getEL(Wi):
        line = lines.Line2D([P[row[0]][0],P[row[1]][0]],[P[row[0]][1],P[row[1]][1]])
        ax.add_line(line)
    for row in SHBi.T:
        row = row.tolist()
        row=row[0]
        plotCycle(ax,row,maximal[0],P)

    plt.show()
    return

def plot_filtration(SHB,Ws,maximal,data):
    for index in range(len(SHB)):
        plot_examples(SHB[index],Ws[index],maximal,P)
    return

def createRandomScatter(N, noise):
    P = []
    for i in range(N):
        x = noise * random()
        y = noise * random()
        P.append([x,y])

    return P

def parallel_pipeline(options):
    GlobalOptions, eps = options
    W = GlobalOptions['Matrix']
    nVert = len(W)
    SHB = []
    Ws = []
    res = {}

    maximal = SimplexGraph.getEdgeList(W)

    stats = []

    Wstep = filterMatrix(W,eps)
    Ws.append(Wstep)

    G = SimplexGraph(nVert,Wstep, maximal)
    cycles = G.getCycleBase()
    #print('trovati i cicli')
    #print("N =",G.NVert,"NEdges =",G.NEdges)
    stats.append( 'NVert = ' + str(G.NVert) + ' NEdges = ' + str(G.NEdges) )
    stats.append( 'Filtration Eps = ' + str(eps) )
    stats.append( 'Keep all Draws = ' + str(GlobalOptions['Draws']) )

    t = time.time()
    d2 = getD2( Wstep , maximal[0] ) # ottiene d2 da Wstep ed edgelist
    #print(d2)
    (An, B1, Z1, H1) = G.getAnnotation(d2,cycles)
    elapsed = time.time()-t
    stats.append( 'H1= '+ str(H1) + ' B1= ' + str(B1) + ' Z1= ' + str(Z1) )
    stats.append( 'Annotazione in sec ' + str(elapsed) )
    #print('fatta annotazione')

    #print('in tempo = ', elapsed)
    #print('H1 = ', H1, ' B1 = ',B1, ' Z1 = ',Z1)
    sup = np.eye( H1, dtype = int )
    Sup = []
    for i in range(H1):
        Sup.append( sup[:,i] )

    G.Support = Sup
    G.fixNTE() # precalcoliamo i non tree edges
    t1 = time.time()
    #print('Ricerca cicli minimi')
    for i in range(H1):
        if GlobalOptions['Draws']:
            listMin = G.findShortestNonOrtho( G.Support[i] , allDraws=True)
            # scegli il primo per l'update
            mincycle = listMin[0][0]
            temp = [x[0] for x in listMin] #leggi riga sotto:
            lMin = np.unique(temp, axis = 0) #IMPO: introdotto per eliminare ripetizioni di cicli
            SHB.append(lMin) # in questo caso avrò una lista di liste invece di una matrice
        else:
            (mincycle, length) = G.findShortestNonOrtho( G.Support[i], allDraws=False )
        G.updateSup(mincycle, i)
        #print('Trovato ciclo ', i , ' in tempo ', elapsed)
        #print( np.matrix(G.Support).transpose())
    elapsed = time.time() -t1
    stats.append( 'Cicli minimi in sec ' + str(elapsed) )
    #print('in tempo = ', elapsed)

    if GlobalOptions['Draws']: # se non voglio tenere i pareggi leggo la matrice da G
        res['SHB'] = SHB
    else:
        res['SHB']=np.matrix(G.SHB).transpose()

    #restituiamo un dizionario
    res['Draws'] = GlobalOptions['Draws']
    res['Filtration Eps'] = eps
    if GlobalOptions['ReturnMatrix']:
        res['Filtered Matrix'] = Ws
    else:
        res['Filtered Matrix'] = None
    if GlobalOptions['ReturnMaximal']:
        res['Max'] = maximal
    else:
        res['Max'] = None
    res['stats'] = stats

    return res

def parallel_rec_pipeline(setup):
    GlobalOptions, eps, connection = setup # if connection is None we ignore it
    W = GlobalOptions['Matrix']
    NV = GlobalOptions['N']
    allDraws = GlobalOptions['Draws']
    filename = GlobalOptions['monitor']
    maximal = GlobalOptions['EdgeList'] # maximal is (edgelist,weights)

    Wstep = filterMatrix(W,eps)
    G = SimplexGraph(NV,Wstep, maximal, connection, eps)
    G.getCycleBase()
    d2 = getD2( Wstep , maximal[0]) # voglio solo edgelist, non i pesi!
    (An, B1, Z1, H1) = G.getAnnotation(d2)

    G.initSup(allDraws) # init SPT
    G.ExtendBasis(1,H1) # Compute SHB!

    return (eps, G.SHB)

def compute_cycles(GlobalOptions, epsList):

    Filt = Geometry.Filtration()
    Parallel = GlobalOptions['parallel']
    monitorFile = GlobalOptions['monitor']

    if Parallel:

        # If necessary setup the monitoring system
        if monitorFile is not None:
            # setup the pipes
            pipes = [ Pipe(duplex=True) for _ in epsList]
            parents = [ e[0] for e in pipes]
            children = [ e[1] for e in pipes]
            del pipes
            # setup the monitor object

            Tracker = Monitor( parents , monitorFile , epsList  )

            setup = zip([GlobalOptions for i in epsList ],epsList, children) # list of tuples ( data , eps, pipe)

        else: # if monitor is none
            setup = zip([GlobalOptions for i in epsList ],epsList, [None for _ in epsList])
            Tracker = None

        pool = Pool(multiprocessing.cpu_count() - 1 )
        job = pool.map_async(parallel_rec_pipeline, setup)

        while job._number_left != 0:
            if Tracker is not None: # if monitoring is active
                if any( [ rec.poll() for rec in parents ] ):
                    Tracker.track()
                    #time.sleep(3)

        # then fetch the result
        res = job.get()

        # JUST IN CASE pull the last updates from the pipes, then close them
        if Tracker is not None: # if monitoring is active
            if any( [ rec.poll() for rec in parents ] ):
                Tracker.track()

            for p,c in zip(parents,children):
                p.close()
                c.close()

        # Now computing the filtration
        for b in res: # add to the filtration
            Filt.add( b[0] , b[1] )

        Filt.sort_by_eps() # sort the Filtration by epsilon

        maximal = GlobalOptions['EdgeList'] # maximal is (edgelist,weights)
        Filt.set_edgeList(maximal[0]) # set edgelist (without weights)

        return Filt

    else: # If not parallel
        W = GlobalOptions['Matrix']
        maximal = GlobalOptions['EdgeList'] # maximal is (edgelist,weights)
        allDraws = GlobalOptions['Draws']

        nVert = len(W)

        res = []

        if not epsList or len(epsList)==0 :
            raise ValueError("Empty List of Thresholds!")

        for e in epsList:

            Wstep = filterMatrix(W,e)

            G = SimplexGraph(nVert,Wstep, maximal)
            G.getCycleBase()
            d2 = getD2( Wstep , maximal[0] ) # qua c'era un bug!! Voglio solo
            # maximal[0], solo gli edges e non i pesi!
            (An, B1, Z1, H1) = G.getAnnotation(d2)

            G.initSup(allDraws) # init SPT
            G.ExtendBasis(1,H1) # Compute SHB!

            Filt.add( e , G.SHB )

            Filt.set_edgeList(maximal[0])

        return Filt

class Monitor(object):
    """
    Instances of this class track the progression of the execution, by collecting
    messages through the pipes and managing it, then printing it to a file.
    Constructor Input:
        - A list of 'receiver' pipes
        - A filename to write the output to
        - The list of epsilon values in the filtration
    Defines methods to poll the pipes, read them if necessary, and then write to file
    """

    def __init__(self, receivers , write_to, epsList):

        self.write_to = write_to

        try:
            self.stream = open(self.write_to, 'w')
        except:
            raise ValueError('Destination file cannot be written!')

        self.receivers = receivers
        self.start = datetime.datetime.now()

        if epsList is None or len(epsList) == 0:
            raise ValueError('Not a correct EpsList!')

        self.epsList = epsList
        self.statuses = [{'started':False, 'eps':eps, 'done':0, 'total':None, \
         'pid':None, 'edges':None, 'start_time':None, 'finished':False, \
         'end_time':None } for eps in self.epsList]

        self.sep = ""
        self.maxlen = 32
        for _ in range(62):
            self.sep += "-"

        import sys
        self.source = sys.argv[0]

        self.myprint()

    def myprint(self):

        print( "          *** Monitor *** ", file=self.stream )
        print( "Start : " + str(self.start) + " - Last Poll : " + str( datetime.datetime.now() ) + \
        " - Elapsed : " + str( datetime.datetime.now() - self.start ), file=self.stream )
        print( "Script: " + str(self.source) + " - PID: " + str(multiprocessing.current_process().pid) + "\n" , file=self.stream)
        string = "{:^5}|{:^9}|{:^8}|{:^15}|{:^10} ({:^6}) |".format("N","EPS","WRKR","ELAPSED","ADV","EDGS")
        print(string, file= self.stream)
        print(self.sep, file= self.stream)

        for i,stat in enumerate(self.statuses):
            # CLUSTERFUUUUCK!
            #string = "{:<4}. Eps {:.3f}".format(i+1, stat['eps'])
            string = "{:<4} . {:>7}".format(i+1, "{:.3f}".format(stat['eps']))

            if not stat['started']: # not yet started
                worker = "{:^6}".format("None")
                done = ""
                tot = ""
                edg = ""
                ss = "{:^4}/{:^4} ({:^6})".format(done,tot,edg)
                elapsed = datetime.timedelta(microseconds=0)
                fraction = 0.0

            elif stat['finished']: # already finished
                worker = "{:^6}".format(str(stat['pid'])) # pid del worker
                done = stat['done']
                tot = stat['total']
                edg = stat['edges']
                try:
                    fraction = float(done)/tot
                except ZeroDivisionError: # homology is trivial, so it's done!
                    fraction = 1.0
                ss = "{:^4}/{:^4} ({:^6})".format(str(done),str(tot),str(edg))
                elapsed = stat['end_time'] - stat['start_time']

            else: # in the process
                worker = "{:^6}".format(str(stat['pid'])) # pid del worker
                done = stat['done']
                elapsed = datetime.datetime.now() - stat['start_time']
                edg = stat['edges']
                if stat['total'] is not None:
                    tot = stat['total']
                    try:
                        fraction = float(done)/tot
                    except ZeroDivisionError: # homology is trivial, so it's done!
                        fraction = 1.0
                    ss = "{:^4}/{:^4} ({:^6})".format(str(done),str(tot),str(edg))
                else: # it hasn't begun yet
                    fraction = 0.0
                    done = ""
                    tot = ""
                    edg = ""
                    ss = "{:^4}/{:^4} ({:^6})".format(done,tot,edg)

            s = elapsed.seconds
            h = s // 3600
            s -= 3600*h
            m = s // 60
            s -= 60*m
            elapsed = "{:^5}:{:^3}:{:^3}".format(str(h)+'h',str(m)+'m',str(s)+'s')

            #string += " - Wrkr: "+worker+" - ET: "+elapsed+" | "+ss+" "+self.bar(fraction)
            string += " - "+worker+" - "+elapsed+" | "+ss+" "+self.bar(fraction)
            if stat['finished']:
                string += "  *  Done! "

            print(string , file=self.stream)

    def bar(self, fraction):
        """
        Write a string of bars to represent advancement
        """
        adv = math.floor(fraction * self.maxlen)
        left = self.maxlen - adv
        string = "|"
        for _ in range(adv):
            string = string + "/"
        for _ in range(left):
            string = string + "-"
        string += "|"
        return string

    def track(self):
        """
        Polls all pipes, retrieves content if any, re-opens file so as to overwrite it,
        and writes to it
        """
        for ind,rec in enumerate(self.receivers):
            if rec.poll():
                msg = rec.recv()
                eps = msg[0]
                pid = msg[1]
                done = msg[2]
                tot = msg[3]
                edg = msg[4]

                if not self.statuses[ind]['started']:
                    self.statuses[ind]['started'] = True
                    self.statuses[ind]['start_time'] = datetime.datetime.now()
                    self.statuses[ind]['total'] = tot
                    self.statuses[ind]['pid'] = pid
                    self.statuses[ind]['edges'] = edg
                    if tot == 0: # homology is trivial
                        self.statuses[ind]['finished'] = True
                        self.statuses[ind]['end_time'] = datetime.datetime.now()
                else:
                    self.statuses[ind]['done'] = done
                    if done == tot: # if finished
                        self.statuses[ind]['finished'] = True
                        self.statuses[ind]['end_time'] = datetime.datetime.now()
        # statuses are updated!

        # re-open file to overwrite it
        self.stream = open(self.write_to, 'w')
        self.myprint()


def getFiltrBasis(W, epsList, Draws=True, parallel = False, monitor = None):
    """ Input a weighted adjacency matrix W
        Output a weighted adjacency matrix S that is the minimal homological scaffold of H_1
    """

    if not epsList or len(epsList) == 0:
        raise ValueError('Empty threshold list')

    NV = len(W)

    maximal = SimplexGraph.getEdgeList(W)

    GlobalOptions = {}
    GlobalOptions['Matrix'] = W
    GlobalOptions['N'] = NV
    GlobalOptions['EdgeList'] = maximal
    GlobalOptions['Draws'] = Draws
    GlobalOptions['parallel'] = parallel
    GlobalOptions['monitor'] = monitor
    GlobalOptions['ReturnMatrix'] = False
    GlobalOptions['ReturnMaximal'] = False

    Filtr = compute_cycles(GlobalOptions, epsList)

    return Filtr
