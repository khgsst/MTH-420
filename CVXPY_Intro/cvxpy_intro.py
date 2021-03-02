# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Labib Zakaria
MTH 420
3/5/21
"""
import numpy as np
import cvxopt as co

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    c=co.matrix(np.array([[2.],[1.],[3.]]))
    G=co.matrix(np.array([[1.,2.,0.],[0.,1.,-4.],[-2.,-10.,-3.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]))
    h=co.matrix(np.array([[3.],[1.],[-12.],[0.],[0.],[0.]]))
    sol=co.solvers.lp(c,G,h)
    print(sol['x'])
    print(sol['primal objective'])
    return sol['x'],sol['primal objective']
    raise NotImplementedError("Problem 1 Inccmplete")

prob1()
# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    (m,n)=A.shape
    P=np.reshape(np.ones(n),(n,1))
    Q=np.reshape(np.zeros(n),(n,1))
    c=co.matrix(np.vstack((P,Q)))
    nI=-1*(np.zeros((n,n))+np.diag(np.diag(np.ones((n,n)))))
    zm=np.zeros((m,n))
    zn=np.zeros((n,n))
    L=np.vstack((nI,nI,nI))
    R=np.vstack((-1*nI,nI,zn))
    G=co.matrix(np.column_stack((L,R)))
    print(G)
    h=co.matrix(np.zeros(3*n))
    A=co.matrix(np.column_stack((zm,A)))
    print(np.vstack((G,A)))
    b=b.astype(np.float)
    b=co.matrix(b)
    sol=co.solvers.lp(c,G,h,A,b)
    sol['x']=sol['x'][n:2*n]
    print(sol['x'])
    print(sol['primal objective'])
    return sol['x'],sol['primal objective']
    raise NotImplementedError("Problem 2 Incomplete")
A=np.array([[1,2,1,1],[0,3,-2,1]])
b=np.array([[7],[4]])
l1Min(A, b)
# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    c=co.matrix(np.array([[4.],[7.],[6.],[8.],[8.],[9.]]))
    nI6=-1*np.eye(6)
    L=np.array([[0,1,0,1,0,1],[0,-1,0,-1,0,-1]])
    G=co.matrix(np.vstack((nI6,L)))
    z6=np.reshape(np.zeros(6),(6,1))
    h=co.matrix(np.vstack((z6,np.array([[8],[-8]]))))
    A=co.matrix(np.array([[1.,1.,0.,0.,0.,0.],[0.,0.,1.,1.,0.,0.],[0.,0.,0.,0.,1.,1.],[1.,0.,1.,0.,1.,0.]]))
    b=co.matrix(np.array([[7.],[2.],[4.],[5.]]))
    sol=co.solvers.lp(c,G,h,A,b)
    print(sol['x'],sol['primal objective'])
    return sol['x'],sol['primal objective']
    raise NotImplementedError("Problem 3 Incomplete")

prob3()
# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q=co.matrix(np.array([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]]))
    r=co.matrix(np.array([[3.],[0.],[1.]]))
    sol=co.solvers.qp(Q,r)
    print(sol['x'],sol['primal objective'])
    return sol['x'],sol['primal objective']
    raise NotImplementedError("Problem 4 Incomplete")

prob4()
# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    (m,n)=A.shape
    Q=co.matrix(2*np.transpose(A.astype(np.float))@A.astype(np.float))
    r=co.matrix(np.reshape(-2*np.reshape(b.astype(np.float),(1,m))@A,(n,1)))
    A0=co.matrix(np.ones((1,n)))
    b0=co.matrix(np.array([1.]))
    G=co.matrix(-1*np.eye(n))
    h=co.matrix(np.zeros((n,1)))
    sol=co.solvers.qp(Q,r,G,h,A0,b0)
    sol['primal objective']=(sol['primal objective']+np.reshape(b.astype(np.float),(1,m))@b.astype(np.float))**(1/2)
    print(sol['x'],sol['primal objective'])
    return sol['x'],sol['primal objective']
    raise NotImplementedError("Problem 5 Incomplete")

A=np.array([[1,2,1,1],[0,3,-2,1]])
b=np.array([[7],[4]])
prob5(A, b)
# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")
