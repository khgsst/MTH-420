# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Labib Zakaria>
<MTH 420>
<1/29/21>
"""
from matplotlib import pyplot as plt
import numpy as np

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    a=np.random.normal(size=(n,n))
    b=np.mean(a,axis=1)
    c=np.var(b)
    print(c)
    return c
    raise NotImplementedError("Problem 1 Incomplete")

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    m=np.linspace(100,1000,10)
    print(m)
    v=np.zeros((10))
    for i in range(0,len(m)):
        v[i]=var_of_means(np.int64(m[i]))
    print(v)
    plt.plot(m,v)
    return plt.show() 
    raise NotImplementedError("Problem 1 Incomplete")
var_of_means(5)
prob1()

# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x=np.linspace(-2*np.pi,np.pi,300)
    plt.plot(x,np.sin(x))
    a=plt.show()
    plt.plot(x,np.cos(x))
    b=plt.show()
    plt.plot(x,np.arctan(x))
    c=plt.show()
    plt.ion()
    plt.plot(x,np.sin(x))
    plt.plot(x,np.cos(x))
    plt.plot(x,np.arctan(x))
    plt.ioff()
    d=plt.show()
    return a,b,c,d
    raise NotImplementedError("Problem 2 Incomplete")

prob2()
# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x=np.linspace(-2,0.99999999,150)
    y=np.linspace(1.00000001,6,150)
    f=1/(x-1)
    g=1/(y-1)
    plt.ion()
    plt.plot(x,f,'m--',linewidth=4,label="1/(x-1),x<1")
    plt.plot(y,g,'m--',linewidth=4,label="1/(x-1),x>1")
    plt.ioff()
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.legend(loc="upper left")
    d=plt.show()
    return d
    raise NotImplementedError("Problem 3 Incomplete")

prob3()
# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x=np.linspace(0,2*np.pi,150)
    a=plt.subplot(2,2,1)
    a.plot(x,np.sin(x),'g-')
    a.set_title("sin(x)", fontsize=18)
    plt.axis([0,2*np.pi,-2,2])
    b=plt.subplot(2,2,2)
    b.plot(x,np.sin(2*x),'r--')
    b.set_title("sin(2x)", fontsize=18)
    plt.axis([0,2*np.pi,-2,2])
    c=plt.subplot(2,2,3)
    c.plot(x,2*np.sin(x),'b--')
    c.set_title("2sin(x)", fontsize=18)
    plt.axis([0,2*np.pi,-2,2])
    d=plt.subplot(2,2,4)
    plt.plot(x,2*np.sin(2*x),'m:')
    d.set_title("2sin(2x)", fontsize=18)
    plt.axis([0,2*np.pi,-2,2])
    plt.suptitle("Variations on sin(x)")
    e=plt.show()
    return e 
    raise NotImplementedError("Problem 4 Incomplete")

prob4()
# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x=np.linspace(-2*np.pi,2*np.pi,300)
    y=x.copy()
    X,Y=np.meshgrid(x,y)
    Z=np.sin(X)*np.sin(Y)/(X*Y)
    a=plt.subplot(121)
    c=a.pcolormesh(X,Y,Z,shading="auto",cmap="viridis")
    plt.colorbar(c)
    a.axis([-2*np.pi,2*np.pi,-2*np.pi,2*np.pi])
    b=plt.subplot(122)
    d=b.contourf(X,Y,Z,20,cmap="coolwarm")
    plt.colorbar(d)
    b.axis([-2*np.pi,2*np.pi,-2*np.pi,2*np.pi])
    c=plt.show()
    return c
    raise NotImplementedError("Problem 6 Incomplete")
prob6()
