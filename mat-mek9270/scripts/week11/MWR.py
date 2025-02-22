import sympy as sp
import numpy as np
from numpy.polynomial import Legendre as Leg
from numpy.polynomial import Chebyshev as Cheb
from scipy.integrate import quad

x = sp.Symbol('x')

class Helmholtz:

    def __init__(self, alpha=1, method: str = 'legendre'):
        self.alpha = alpha
        self.u = None # Manufactored solution
        self.bc = (0,0)
        self.basis = None

    def stiff_matrix(self):
        S = np.zeros((self.N+1,self.N+1))
        for i in range(self.N+1):
            for j in range(self.N+1):
                P_ddP = lambda x_i: self.basis(i)(x_i) * self.basis(j).deriv(2)(x_i)
                P_P = lambda x_i: self.basis(i)(x_i) * self.basis(j)(x_i)
                S[i,j] = quad(P_ddP,-1,1)[0] + self.alpha*quad(P_P, -1, 1)[0]
        return S

    def Boundary_func(self,xi):
        return self.bc[0]/2*(1-xi) + self.bc[1]/2*(1+xi)

    def b(self):
        b = np.zeros(self.N+1)
        f = sp.diff(self.u, x, 2) + self.alpha*self.u
        for i in range(self.N+1):
            P_f = lambda x_i: self.basis(i)(x_i) * (f.subs(x,x_i)-self.alpha*self.Boundary_func(x_i))
            b[i] = quad(P_f,-1,1)[0]

        return b

    def l2(self, uN, uE, xi):
        return np.trapz(np.sqrt((uE-uN)**2),xi)

    def uN(self, u_coeffs, xi):
        uN = 0
        for i, u_ in enumerate(u_coeffs):
            uN += self.basis(i)(xi)*u_
        uN += self.Boundary_func(xi)
        return uN
            
    def __call__(self, N, alpha, u):
        x_ = np.linspace(-1,1,N+1)
        self.N = N
        self.alpha = 0.1
        self.u = u
        self.bc=(self.u.subs(x,-1),self.u.subs(x,1))
        self.basis = lambda i: Leg.basis(i) - Leg.basis(i+2)
        A = self.stiff_matrix()
        b = self.b()
        u_coeffs = np.linalg.solve(A,b)
        uN_arr = np.zeros(self.N+1)
        for i, xi in enumerate(x_):
            uN_arr[i] = self.uN(u_coeffs, xi)

        uE_arr = sp.lambdify(x,u)(x_)
        l2 = self.l2(uN_arr,uE_arr,x_)
        
        return uN_arr, uE_arr, x_, l2
        
        
    
    
    

