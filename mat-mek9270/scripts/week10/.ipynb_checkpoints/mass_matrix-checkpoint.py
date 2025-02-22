import numpy as np
import sympy as sp
from scipy.integrate import quad


x = sp.Symbol('x')
h = sp.Symbol('h')

def Lagrangebasis(xj, x=x):
    """Construct Lagrange basis for points in xj

    Parameters
    ----------
    xj : array
        Interpolation points (nodes)
    x : Sympy Symbol

    Returns
    -------
    Lagrange basis as a list of Sympy functions
    """
    n = len(xj)
    ell = []
    numert = sp.Mul(*[x - xj[i] for i in range(n)])
    for i in range(n):
        numer = numert/(x - xj[i])
        denom = sp.Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
        ell.append(numer/denom)
    return ell
    
def Ade(d=1):
    xj = np.linspace(-1,1,d+1)
    lbasis = Lagrangebasis(xj)
    ae = lambda r, s: sp.integrate(lbasis[r]*lbasis[s], (x, -1, 1))
    Ae = np.zeros((d+1,d+1))
    #calculate upper diagonal matrix
    for i in range(d+1):
        for j in range(i,d+1):
                Ae[i,j] = ae(i,j)
                Ae[j,i] = Ae[i,j]
    """
    #set the lower diagonal equal to the same values as the upper
    Ae += Ae.T
    #calculate the main diagonal
    for i in range(d+1):
        Ae[i,i] = ae(i,i)
    """
    Ae = h/2*sp.Matrix(Ae)
    return Ae

def b(u, xj, d=1):
    """u is sympy function of x"""
    N_nodes = len(xj) - 1
    N_elements = N_nodes//d # // = floor division
    
    b = np.zeros(N_nodes+1)
    xi = np.linspace(-1,1,d+1)
    lbasis = Lagrangebasis(xi)

    for elem in range(N_elements):
        xL, xR = get_element_boundaries(xj, elem, d=d)
        hj = xR-xL
        xM = 1/2*(xR+xL)
        X = xM+hj/2*x #counter-intuitive alert: X = our real varibale. little x = our mapped variable (-1 -> 1).
        u_FOR_THIS_SPECIFIC_ELEMENT = u.subs(x,X)
        for i in range(d+1):
            be_f = lambda Xk: sp.lambdify(x,u_FOR_THIS_SPECIFIC_ELEMENT)(Xk) * sp.lambdify(x,lbasis[i])(Xk)
            integral = quad(be_f,-1,1)
            b[local_to_global_map(e=elem, r=i, d=d)] += hj/2*integral[0]
            
    return b
                      
def get_element_boundaries(xj, e, d=1):
    return xj[d*e], xj[d*(e+1)]

def get_element_length(xj, e, d=1):
    xL, xR = get_element_boundaries(xj, e, d=d)
    return xR-xL

def local_to_global_map(e, r=None, d=1): # q(e, r)
    if r is None:
        return slice(d*e, d*e+d+1) # = (d*e):(d*e+d+1)
    return d*e+r

def assemble_mass(xj, d=1):
    N_nodes = len(xj) - 1
    N_elements = N_nodes//d
    A = np.zeros((N_nodes+1, N_nodes+1))
    Ad = Ade(d)
    for elem in range(N_elements):
        hj = get_element_length(xj, elem, d=d)
        s0 = local_to_global_map(elem, d=d)
        A[s0, s0] += np.array(Ad.subs(h, hj), dtype=float) #Set A[(i,i+1/2/...),(i,i+1/2/3...)] elements 
    
    return A



