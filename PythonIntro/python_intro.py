# python_intro.py
"""Python Essentials: Introduction to Python.
<Labib Zakaria>
<MTH 420>
<1/22/21>
"""
# Problem 1
name="Labib Zakaria"
if name=="Labib Zakaria":
    print("Hello, world!")
# Problem 2
def sphere_volume(r):
    """Return the volume of a sphere with specified radius"""
    return (4*3.14159*r**3)/3
if name=="Labib Zakaria":
    print(sphere_volume(1))
# Problem 3
def isolate(a,b,c,d,e):
    """Print a,b,c with 5 spaces of separation between & c,d,e with 1 space of separation between"""
    L=print(a,b,c,sep='     ',end=' ')
    print(d,e,sep=' ')
    return L
isolate(1,2,3,4,5)
# Problem 4
def first_half(word):
    """Display the first half of the input string (excluding the middle character) if it exists"""
    return word[:len(word)//2]
first_half("it")
first_half("toot")
def backward(word):
    """Display the input string backwards"""
    return word[::-1]
backward("nim")
# Problem 5
def list_ops():
    """Apply list operations & display resulting list"""
    # 0: Define a list with the entries "bear", "ant", "cat", and "dog", in that order 
    l=["bear","ant","cat","dog"]
    print(l)
    # 1: Append "eagle"
    l.append("eagle")
    print(l)
    # 2: Replace the entry at index 2 with "fox"
    l[2]="fox"
    print(l)
    # 3: Remove (or pop) the entry at index 1
    l.remove(l[1])
    print(l)
    # 4: Sort the list in reverse alphabetical order
    l.sort(reverse=True)
    print(l)
    # 5: Replace "eagle" with "hawk"
    l[l.index("eagle")]="hawk"
    print(l)
    # 6: Add the string "hunter" to the last entry in the list
    l[len(l)-1]=l[len(l)-1]+"hunter"
    return l
list_ops()
# Problem 6
def pig_latin(word):
    """Translate given word into Pig Latin"""
    vowels=('a','e','i','o','u')
    if word.lower().startswith(vowels):
       r=word.lower()+"hay"
    else:
       r=word[1:]+word[0].lower()+"ay"
    return r                
pig_latin("Spell" )
pig_latin("it")
# Problem 7
def palindrome():
    """Find the largest palindrome that is a product of three digit natural numbers"""
    for v in range(999,99,-1):
        for t in range(v,99,-1):
            u=v*t
            q=str(u)
            if q==q[::-1]:
               return u
palindrome()    
# Problem 8
def alt_harmonic(n):
    """Find the sum of the first n terms of the alternating harmonic sequence/nth partial sum of the alternating harmonic series"""
    s=sum([(-1)**(i+1)/(i) for i in range(1,n+1)])
    return s
alt_harmonic(3)
alt_harmonic(500000)

