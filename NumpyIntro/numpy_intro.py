# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Labib Zakaria
MTH 420
1/29/21
"""
import numpy as np


def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A=np.array([[3,-1,4],[1,5,-9]])
    B=np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])
    AB=A@B
    print(AB)
    return AB 
    raise NotImplementedError("Problem 1 Incomplete")
prob1()
def prob2():
    A=np.array([[3,1,4],[1,5,9],[-5,3,1]])
    C=-A@A@A+9*A@A-15*A
    print(C)
    return C
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    raise NotImplementedError("Problem 2 Incomplete")

prob2()
def prob3():
    A=np.triu(np.ones((7,7)))
    B=np.tril(-1*np.ones((7,7)))+np.triu(5*np.ones((7,7)))
    P=A@B@A
    P=P.astype(np.int64)
    print(P)
    return P
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    raise NotImplementedError("Problem 3 Incomplete")

prob3()
def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B=np.copy(A)
    B[B<0]=0
    print(B)
    return B
    raise NotImplementedError("Problem 4 Incomplete")
A=np.array([[1,2],[-1,-2]])
prob4(A)
def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A=np.vstack((np.linspace(0,4,3),np.linspace(1,5,3)))
    print(A)
    B0=3*np.ones((3,3))
    B=np.tril(B0)
    print(B)
    C0=-2*np.ones((3,3))
    C=np.diag(np.diag(C0))
    print(C)
    D1=np.hstack((np.zeros((3,3)),np.transpose(A),np.ones((3,3))))
    D2=np.hstack((A,np.zeros((2,5))))
    D3=np.hstack((B,np.zeros((3,2)),C))
    D=np.vstack((D1,D2,D3))
    print(D)
    return D 
    raise NotImplementedError("Problem 5 Incomplete")

prob5()
def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.
    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    B=np.diag(1/np.sum(A,axis=1))
    print(B)
    A=B@A
    print(A)
    return A
    raise NotImplementedError("Problem 6 Incomplete")
A = np.array([[1,1,0],[0,1,0],[1,1,1]])
prob6(A)
def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    raise NotImplementedError("Problem 7 Incomplete")
