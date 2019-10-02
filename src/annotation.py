import numpy as np

def annotation(d2,Z):
    """
    d2: border matrix of the set of faces of the simplicial complex, borders are written as rows.
    Z: basis of 1-dim-cycles. Each row is a cycle obtained from a spanning tree and a sentinel edge.
    returns an annotation of sentinel edges. Non sentinel edges have annotation equal to 0.
    """
	
	def low(col):
    	"""
	    col: 1-dimensional array.
    	gets the index of the "lowest" element in col different from 0.
    	if col=0 then low = -1
    	"""
    	l=-1;
    	for i in range(len(col)):
    	    if col[i]>0:
    	        l=i
    	return l

    lowSet = {}; #dictionary with low indexes and relative rows
    i=0;
    while i != len(d2):
        lowRowi=low(d2[i])
        while lowRowi in lowSet.keys():
            d2[i]=(d2[i]+d2[lowSet[lowRowi]])%2
            lowRowi=low(d2[i])
        if lowRowi > -1 :
            lowSet[lowRowi]=i
            i=i+1
        else:
            d2=np.delete(d2,(i),axis=0)
    dimB1=i # dimensione dello spazio dei bordi

    Zt=np.concatenate((d2,Z),axis=0) #we start the reduction from row dimB1
    totRow=len(Zt)
    reductionMatrix=np.identity(totRow,dtype=int)
    Id=np.identity(totRow,dtype=int)
    elementsToDelete=[]

    while i != totRow:
        lowRowi=low(Zt[i])
        while lowRowi in lowSet.keys():
            Zt[i]=(Zt[i]+Zt[lowSet[lowRowi]])%2
            reductionMatrix[i]=(reductionMatrix[i]+Id[lowSet[lowRowi]])%2

            lowRowi=low(Zt[i])
            
        if lowRowi > -1 :
            lowSet[lowRowi]=i
            i=i+1
        else:
            elementsToDelete.append(i)
            i=i+1

    #eliminate coordinates of cycles that are borders:
    reductionMatrix=np.delete(reductionMatrix,elementsToDelete,axis=1); 
    reductionMatrix=np.delete(reductionMatrix,range(dimB1),axis=1); 
    A=np.delete(reductionMatrix,range(dimB1),axis=0);
    """observation: the number of rows of A is the dimension of the 1-dim-cycle group;
       the number of columns is the dimension of the 1st homology group
    """
	A = np.array(A)
	dimZ1 = np.shape(A)[0]
	dimH1 = np.shape(A)[1]
    return (A, dimB1, dimZ1, dimH1)


