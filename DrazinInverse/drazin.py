# drazin.py
"""Volume 1: The Drazin Inverse.
<Labib Zakaria>
<MTH 420>
<2/12/21>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse.csgraph import laplacian as lap
from numpy import linalg as nla


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    b=np.allclose(A@Ad,Ad@A) and np.allclose(nla.matrix_power(A,k+1)@Ad,nla.matrix_power(A,k)) and np.allclose(Ad@A@Ad,Ad)
    print(b)
    return b
    raise NotImplementedError("Problem 1 Incomplete")

A=np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
Ad=np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
k=1
is_drazin(A,Ad,k)
A=np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
Ad=np.array([[0,0,0],[0,0,0],[0,0,0]])
k=3
is_drazin(A,Ad,k)
# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    (n,n)=A.shape    
    f=lambda x:abs(x)>tol
    T1,Q1,k1=la.schur(A,sort=f)
    g=lambda x:abs(x)<=tol
    T2,Q2,k2=la.schur(A,sort=g)
    U=np.column_stack((Q1[:,:k1],Q2[:,:(n-k1)]))
    UI=la.inv(U)
    V=UI@A@U
    Z=np.zeros((n,n))
    if k1!=0:
        MI=la.inv(V[:k1,:k1])
        Z[:k1,:k1]=MI
    P=U@Z@UI
    print("A^D="+str(P))
    return P 
    raise NotImplementedError("Problem 2 Incomplete")

A=np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
Ad=drazin_inverse(A, tol=1e-4)
k=1
is_drazin(A,Ad,k)
A=np.array([[1,1,3],[5,2,6],[-2,-1,-3]])
Ad=drazin_inverse(A, tol=1e-4)
k=3
is_drazin(A,Ad,k)
# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    (n,n)=A.shape
    R=np.zeros((n,n))
    print("R="+str(R))
    L=lap(A)+np.ones((n,n))/n
    print("L="+str(L))
    LD=drazin_inverse(L, tol=1e-4)
    print("L^D="+str(LD))
    for i in range(n):
        for j in range(n):
            R[i,j]=LD[i,i]+LD[j,j]-2*LD[i,j]
        print("R0i="+str(R))
        R[i,i]=0
        print("Ri="+str(R))
    print("R="+str(R))
    return R 
    raise NotImplementedError("Problem 3 Incomplete")
A=np.array([[0,1],[1,0]])
effective_resistance(A)
A=np.array([[0,2],[2,0]])
effective_resistance(A)
A=np.array([[0,1,1],[1,0,1],[1,1,0]])
effective_resistance(A)
A=np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
effective_resistance(A)
# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        raise NotImplementedError("Problem 4 Incomplete")


def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")
def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")
