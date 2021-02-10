# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Labib Zakaria>
<MTH 420>
<2/5/21>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import linalg as la


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R=la.qr(A,mode="economic")
    x=la.solve_triangular(R,np.transpose(Q)@b)
    print(Q,R,x)
    return x
    raise NotImplementedError("Problem 1 Incomplete")
A=np.array([[np.pi**2,-np.pi,1],[(np.pi/2)**2,-np.pi/2,1],[0,0,1],[(np.pi/2)**2,np.pi/2,1],[np.pi**2,np.pi,1]])
b=np.array([[-1],[0],[1],[0],[-1]])
least_squares(A, b)
# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    z=np.load('housing.npy')
    print(z)
    x1=z[:,0]
    y1=z[:,1]
    A=x1.reshape(-1,1)
    b=y1.reshape(-1,1)
    print(A,b)
    c=np.ones((len(A),1))
    A=np.column_stack((A,c))
    print(A)
    x=least_squares(A,b)
    print(x)
    plt.ion()
    a=plt.scatter(x1,y1)
    b=plt.plot(x[0]*x1+x[1])
    plt.ioff()
    e=plt.show()
    return x,a,b,e
    raise NotImplementedError("Problem 2 Incomplete")

line_fit()
# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    z=np.load('housing.npy')
    x1=z[:,0]
    y1=z[:,1]
    A3=np.vander(x1,4)
    A6=np.vander(x1,7)
    A9=np.vander(x1,10)
    A12=np.vander(x1,13)
    x3=least_squares(A3,y1.reshape(-1,1))
    x6=least_squares(A6,y1.reshape(-1,1))
    x9=least_squares(A9,y1.reshape(-1,1))
    x12=least_squares(A12,y1.reshape(-1,1))
    q3=A3@x3
    q6=A6@x6
    q9=A9@x9
    q12=A12@x12
    print(A3,A6,A9,A12,x3,x6,x9,x12,q3,q6,q9,q12)
    plt.subplot(2,2,1)
    a=plt.scatter(x1,y1)
    p3=plt.plot(x1,q3)
    plt.subplot(2,2,2)
    a=plt.scatter(x1,y1)
    p6=plt.plot(x1,q6)
    plt.subplot(2,2,3)
    a=plt.scatter(x1,y1)
    p9=plt.plot(x1,q9)
    plt.subplot(2,2,4)
    a=plt.scatter(x1,y1)
    p12=plt.plot(x1,q12)
    e=plt.show()
    return A3,A6,A9,A12,x3,x6,x9,x12,p3,p6,p9,p12,e
    raise NotImplementedError("Problem 3 Incomplete")

polynomial_fit()
def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")
