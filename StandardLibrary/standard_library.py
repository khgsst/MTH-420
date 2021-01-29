# standard_library.py
"""Python Essentials: The Standard Library.
Labib Zakaria
MTH 420
1/22/21
"""
import calculator as cal
from itertools import chain,combinations

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L),max(L),sum(L)/len(L)
    raise NotImplementedError("Problem 1 Incomplete")
prob1((1,2,3))


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    t1="int"
    t2="str"
    t3="list"
    t4="tuple"
    t5="set"
    int1=1
    int2=int1
    int2=int1+1
    if int2==int1:
        r1=t1+"mutable"
    else:
        r1=t1+"immutable"
    str1="1"    
    str2=str1
    str2=str1+"1"
    if str2==str1:
        r2=t2+"mutable"
    else:
        r2=t2+"immutable"
    list1=[1]
    list2=list1
    list2[0]=2
    if list1==list2:
        r3=t3+"mutable"
    else:
        r3=t3+"immutable"
    tuple1=(0,0)
    my_tuple=tuple1
    my_tuple+=(1,)
    if my_tuple==tuple1:
        r4=t4+"mutable"
    else:
        r4=t4+"immutable"
    set1={1}
    set2=set1
    set2|={2}
    if set2==set1:
        r5=t5+"mutable"
    else:
        r5=t5+"immutable"
    return print(int1,int2,str1,str2,list1,list2,tuple1,my_tuple,set1,set2,r1,r2,r3,r4,r5)
    raise NotImplementedError("Problem 2 Incomplete")
prob2()

# Problem 3
def hypot(a, b):
    a2=cal.prod(a,a)
    b2=cal.prod(b,b)
    z=cal.sum(a2,b2)
    c=cal.sqrt(z)
    return c
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    raise NotImplementedError("Problem 3 Incomplete")

hypot(3,4)
# Problem 4
def power_set(A):
    l=[]
    for z in chain.from_iterable(combinations(A, r) for r in range(1,len(A)+1)):
        y=set(z)
        l.append(y)
    return l
    """Use itertools to compute the power set of A.
    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    raise NotImplementedError("Problem 4 Incomplete")

power_set({1,2,5})
# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
