import numpy as np                   # Import numpy for numerical operations
import sympy as sp                   # Import sympy for symbolic operations
import scipy.sparse as sparse        # Import scipy.sparse for sparse matrix operations
from scipy.integrate import quad     # Import quad for numerical integration
from numpy.polynomial import Legendre as Leg  # Import Legendre polynomials
from numpy.polynomial import Chebyshev as Cheb # Import Chebyshev polynomials

x = sp.Symbol('x')                   # Define symbolic variable x

# Map from the true domain to the reference domain
def map_reference_domain(x, d, r):
    return r[0] + (r[1]-r[0])*(x-d[0])/(d[1]-d[0])

# Map from the reference domain to the true domain
def map_true_domain(x, d, r):
    return d[0] + (d[1]-d[0])*(x-r[0])/(r[1]-r[0])

# Map an expression from the reference domain to the true domain
def map_expression_true_domain(u, x, d, r):
    if d != r:                       # Check if the domains are different
        u = sp.sympify(u)            # Convert u to a sympy expression if it's not
        xm = map_true_domain(x, d, r) # Compute the mapping
        u = u.replace(x, xm)         # Replace x in u with the mapped variable
    return u                         # Return the mapped expression

# Base class for function spaces
class FunctionSpace:
    def __init__(self, N, domain=(-1, 1)):
        self.N = N                   # Number of basis functions
        self._domain = domain        # The domain of the function space

    @property
    def domain(self):
        return self._domain          # Return the domain

    @property
    def reference_domain(self):
        raise RuntimeError           # This should be overridden in derived classes

    @property
    def domain_factor(self):
        d = self.domain              # Get the domain
        r = self.reference_domain    # Get the reference domain
        return (d[1]-d[0])/(r[1]-r[0]) # Compute the domain factor

    def mesh(self, N=None):
        d = self.domain              # Get the domain
        n = N if N is not None else self.N # Set number of mesh points
        return np.linspace(d[0], d[1], n+1) # Generate a linearly spaced mesh

    def weight(self, x=x):
        return 1                     # Default weight function

    def basis_function(self, j, sympy=False):
        raise RuntimeError           # This should be overridden in derived classes

    def derivative_basis_function(self, j, k=1):
        raise RuntimeError           # This should be overridden in derived classes

    def evaluate_basis_function(self, Xj, j):
        return self.basis_function(j)(Xj) # Evaluate the j-th basis function at Xj

    def evaluate_derivative_basis_function(self, Xj, j, k=1):
        return self.derivative_basis_function(j, k=k)(Xj) # Evaluate the j-th derivative basis function at Xj

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)       # Ensure xj is at least 1D
        Xj = map_reference_domain(xj, self.domain, self.reference_domain) # Map xj to reference domain
        P = self.eval_basis_function_all(Xj) # Evaluate all basis functions at Xj
        return P @ uh                # Compute the linear combination

    def eval_basis_function_all(self, Xj):
        P = np.zeros((len(Xj), self.N+1)) # Initialize the matrix
        for j in range(self.N+1):         # Loop through each basis function
            P[:, j] = self.evaluate_basis_function(Xj, j) # Evaluate basis function and store
        return P                         # Return the matrix

    def eval_derivative_basis_function_all(self, Xj, k=1):
        raise NotImplementedError       # This should be overridden in derived classes

    def inner_product(self, u):
        us = map_expression_true_domain(
            u, x, self.domain, self.reference_domain) # Map expression to true domain
        us = sp.lambdify(x, us)         # Convert sympy expression to a function
        uj = np.zeros(self.N+1)         # Initialize the result array
        h = self.domain_factor          # Get the domain factor
        r = self.reference_domain       # Get the reference domain
        for i in range(self.N+1):       # Loop through each basis function
            psi = self.basis_function(i) # Get the i-th basis function
            def uv(Xj): return us(Xj) * psi(Xj) # Define the integrand
            uj[i] = float(h) * quad(uv, float(r[0]), float(r[1]))[0] # Perform the integration
        return uj                       # Return the result array

    def mass_matrix(self):
        return assemble_generic_matrix(TrialFunction(self), TestFunction(self)) # Assemble mass matrix

# Legendre function space
class Legendre(FunctionSpace):

    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain) # Initialize using base class

    def basis_function(self, j, sympy=False):
        if sympy:                   # Check if symbolic form is requested
            return sp.legendre(j, x) # Return sympy legendre polynomial
        return Leg.basis(j)         # Return numerical legendre polynomial

    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k) # Compute the k-th derivative

    def L2_norm_sq(self, N):
        raise NotImplementedError   # Not implemented yet

    def mass_matrix(self):
        raise NotImplementedError   # Not implemented yet

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)      # Ensure xj is at least 1D
        Xj = map_reference_domain(xj, self.domain, self.reference_domain) # Map xj to reference domain
        return np.polynomial.legendre.legval(Xj, uh) # Evaluate the Legendre polynomials

# Chebyshev function space
class Chebyshev(FunctionSpace):

    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain) # Initialize using base class

    def basis_function(self, j, sympy=False):
        if sympy:                   # Check if symbolic form is requested
            return sp.cos(j*sp.acos(x)) # Return sympy Chebyshev polynomial
        return Cheb.basis(j)        # Return numerical Chebyshev polynomial

    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k) # Compute the k-th derivative

    def weight(self, x=x):
        return 1/sp.sqrt(1-x**2)      # Chebyshev-specific weight function

    def L2_norm_sq(self, N):
        raise NotImplementedError    # Not implemented yet

    def mass_matrix(self):
        raise NotImplementedError    # Not implemented yet

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)       # Ensure xj is at least 1D
        Xj = map_reference_domain(xj, self.domain, self.reference_domain) # Map xj to reference domain
        return np.polynomial.chebyshev.chebval(Xj, uh) # Evaluate the Chebyshev polynomials

    def inner_product(self, u):
        us = map_expression_true_domain(
            u, x, self.domain, self.reference_domain) # Map expression to true domain
        # change of variables to x=cos(theta)
        us = sp.simplify(us.subs(x, sp.cos(x)), inverse=True) # Simplify expression with x=cos(theta)
        us = sp.lambdify(x, us)         # Convert sympy expression to a function
        uj = np.zeros(self.N+1)         # Initialize the result array
        h = float(self.domain_factor)   # Get the domain factor
        k = sp.Symbol('k')              # Define symbolic variable k
        basis = sp.lambdify((k, x), sp.simplify(
            self.basis_function(k, True).subs(x, sp.cos(x), inverse=True))) # Lambdify the basis function
        for i in range(self.N+1):       # Loop through each basis function
            def uv(Xj, j): return us(Xj) * basis(j, Xj) # Define the integrand
            uj[i] = float(h) * quad(uv, 0, np.pi, args=(i,))[0] # Perform the integration
        return uj                       # Return the result array

# Base class for trigonometric function spaces
class Trigonometric(FunctionSpace):
    """Base class for trigonometric function spaces"""

    @property
    def reference_domain(self):
        return (0, 1)                  # Trigonometric reference domain

    def mass_matrix(self):
        return sparse.diags([self.L2_norm_sq(self.N+1)], [0], (self.N+1, self.N+1), format='csr') # Construct mass matrix

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)         # Ensure xj is at least 1D
        Xj = map_reference_domain(xj, self.domain, self.reference_domain) # Map xj to reference domain
        P = self.eval_basis_function_all(Xj) # Evaluate all basis functions at Xj
        return P @ uh + self.B.Xl(Xj)  # Compute linear combination and add boundary condition

# Sines function space
class Sines(Trigonometric):

    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        Trigonometric.__init__(self, N, domain=domain) # Initialize using base class
        self.B = Dirichlet(bc, domain, self.reference_domain) # Apply Dirichlet boundary condition

    def basis_function(self, j, sympy=False):
        if sympy:                     # Check if symbolic form is requested
            return sp.sin((j+1)*sp.pi*x) # Return sympy sine function
        return lambda Xj: np.sin((j+1)*np.pi*Xj) # Return numerical sine function

    def derivative_basis_function(self, j, k=1):
        scale = ((j+1)*np.pi)**k * {0: 1, 1: -1}[(k//2) % 2] # Compute scaling factor for derivative
        if k % 2 == 0:                # Check if k is even
            return lambda Xj: scale*np.sin((j+1)*np.pi*Xj) # Return derivative for even k
        else:
            return lambda Xj: scale*np.cos((j+1)*np.pi*Xj) # Return derivative for odd k

    def L2_norm_sq(self, N):
        return 0.5                    # L2 norm squared

# Cosines function space (currently not implemented)
class Cosines(Trigonometric):

    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        raise NotImplementedError     # Not implemented yet

    def basis_function(self, j, sympy=False):
        raise NotImplementedError     # Not implemented yet

    def derivative_basis_function(self, j, k=1):
        raise NotImplementedError     # Not implemented yet

    def L2_norm_sq(self, N):
        raise NotImplementedError     # Not implemented yet

# Classes to hold the boundary function
class Dirichlet:

    def __init__(self, bc, domain, reference_domain):
        d = domain                   # Get the domain
        r = reference_domain         # Get the reference domain
        h = d[1]-d[0]                # Compute the domain interval
        self.bc = bc                 # Boundary conditions
        self.x = bc[0]*(d[1]-x)/h + bc[1]*(x-d[0])/h # Compute the boundary function in physical coordinates
        self.xX = map_expression_true_domain(self.x, x, d, r)  # Compute the boundary function in reference coordinates
        self.Xl = sp.lambdify(x, self.xX) # Lambdify the boundary function

class Neumann:

    def __init__(self, bc, domain, reference_domain):
        d = domain                   # Get the domain
        r = reference_domain         # Get the reference domain
        h = d[1]-d[0]                # Compute the domain interval
        self.bc = bc                 # Boundary conditions
        self.x = bc[0]/h*(d[1]*x-x**2/2) + bc[1]/h*(x**2/2-d[0]*x)  # Compute the boundary function in physical coordinates
        self.xX = map_expression_true_domain(self.x, x, d, r)       # Compute the boundary function in reference coordinates
        self.Xl = sp.lambdify(x, self.xX) # Lambdify the boundary function

# Base class for function spaces created as linear combinations of orthogonal basis functions
class Composite(FunctionSpace):
    """Base class for function spaces created as linear combinations of orthogonal basis functions

    The composite basis functions are defined using the orthogonal basis functions
    (Chebyshev or Legendre) and a stencil matrix S. The stencil matrix S is used
    such that basis function i is

    .. math::

        \psi_i = \sum_{j=0}^N S_{ij} Q_j

    where :math:`Q_i` can be either the i'th Chebyshev or Legendre polynomial

    For example, both Chebyshev and Legendre have Dirichlet basis functions

    .. math::

        \psi_i = Q_i-Q_{i+2}

    Here the stencil matrix will be

    .. math::

        s_{ij} = \delta_{ij} - \delta_{i+2, j}, \quad (i, j) \in (0, 1, \ldots, N) \times (0, 1, \ldots, N+2)

    Note that the stencil matrix is of shape :math:`(N+1) \times (N+3)`.
    """

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)       # Ensure xj is at least 1D
        Xj = map_reference_domain(xj, self.domain, self.reference_domain) # Map xj to reference domain
        P = self.eval_basis_function_all(Xj) # Evaluate all basis functions at Xj
        return P @ uh + self.B.Xl(Xj) # Compute linear combination and add boundary condition

    def mass_matrix(self):
        M = sparse.diags([self.L2_norm_sq(self.N+3)], [0],
                         shape=(self.N+3, self.N+3), format='csr') # Construct mass matrix
        return self.S @ M @ self.S.T  # Compute composite mass matrix

# Dirichlet Legendre composite function space
class DirichletLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Legendre.__init__(self, N, domain=domain) # Initialize Legendre part
        self.B = Dirichlet(bc, domain, self.reference_domain) # Apply Dirichlet boundary condition
        self.S = sparse.diags((1, -1), (0, 2), shape=(N+1, N+3), format='csr') # Create stencil matrix

    def basis_function(self, j, sympy=False):
        raise NotImplementedError     # Not implemented yet

# Neumann Legendre composite function space (currently not implemented)
class NeumannLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        raise NotImplementedError     # Not implemented yet

    def basis_function(self, j, sympy=False):
        raise NotImplementedError     # Not implemented yet

# Dirichlet Chebyshev composite function space
class DirichletChebyshev(Composite, Chebyshev):

    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Chebyshev.__init__(self, N, domain=domain) # Initialize Chebyshev part
        self.B = Dirichlet(bc, domain, self.reference_domain) # Apply Dirichlet boundary condition
        self.S = sparse.diags((1, -1), (0, 2), shape=(N+1, N+3), format='csr') # Create stencil matrix

    def basis_function(self, j, sympy=False):
        if sympy:                   # Check if symbolic form is requested
            return sp.cos(j*sp.acos(x)) - sp.cos((j+2)*sp.acos(x)) # Return sympy composite basis function
        return Cheb.basis(j)-Cheb.basis(j+2) # Return numerical composite basis function

# Neumann Chebyshev composite function space (currently not implemented)
class NeumannChebyshev(Composite, Chebyshev):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        raise NotImplementedError     # Not implemented yet

    def basis_function(self, j, sympy=False):
        raise NotImplementedError     # Not implemented yet

# Class representing a basis function
class BasisFunction:

    def __init__(self, V, diff=0, argument=0):
        self._V = V                  # Function space
        self._num_derivatives = diff # Number of derivatives
        self._argument = argument    # Argument index

    @property
    def argument(self):
        return self._argument        # Return argument index

    @property
    def function_space(self):
        return self._V               # Return function space

    @property
    def num_derivatives(self):
        return self._num_derivatives # Return number of derivatives

    def diff(self, k):
        return self.__class__(self.function_space, diff=self.num_derivatives+k) # Return differentiated function

# Class representing a test function (inherits from BasisFunction)
class TestFunction(BasisFunction):

    def __init__(self, V, diff=0):    # Initialization method
        BasisFunction.__init__(self, V, diff=diff, argument=0)  # Initialize using base class with argument=0

# Class representing a trial function (inherits from BasisFunction)
class TrialFunction(BasisFunction):

    def __init__(self, V, diff=0):    # Initialization method
        BasisFunction.__init__(self, V, diff=diff, argument=1)  # Initialize using base class with argument=1

# Function to assemble a generic matrix based on trial and test functions
def assemble_generic_matrix(u, v):
    assert isinstance(u, TrialFunction)  # Check if u is an instance of TrialFunction
    assert isinstance(v, TestFunction)   # Check if v is an instance of TestFunction
    V = v.function_space                 # Get the function space from the test function
    assert u.function_space == V         # Ensure both functions belong to the same space
    r = V.reference_domain               # Get the reference domain of the function space
    D = np.zeros((V.N+1, V.N+1))         # Initialize a zero matrix for the result
    cheb = V.weight() == 1/sp.sqrt(1-x**2) # Check if the weight function is Chebyshev-type
    symmetric = True if u.num_derivatives == v.num_derivatives else False # Determine if the matrix is symmetric
    w = {'weight': 'alg' if cheb else None,
         'wvar': (-0.5, -0.5) if cheb else None} # Integration weight parameters
    def uv(Xj, i, j): return (V.evaluate_derivative_basis_function(Xj, i, k=v.num_derivatives) *
                              V.evaluate_derivative_basis_function(Xj, j, k=u.num_derivatives)) # Define the integrand
    for i in range(V.N+1):               # Loop over all basis functions
        for j in range(i if symmetric else 0, V.N+1): # Loop over basis functions (consider symmetry)
            D[i, j] = quad(uv, float(r[0]), float(r[1]), args=(i, j), **w)[0] # Perform integration
            if symmetric:                # If matrix is symmetric
                D[j, i] = D[i, j]        # Set the symmetric element
    return D                             # Return the assembled matrix

# Function to compute the inner product of two functions
def inner(u, v: TestFunction):
    V = v.function_space                # Get the function space from the test function
    h = V.domain_factor                 # Get the domain factor
    if isinstance(u, TrialFunction):    # Check if u is a TrialFunction
        num_derivatives = u.num_derivatives + v.num_derivatives # Sum the number of derivatives
        if num_derivatives == 0:        # If no derivatives
            return float(h) * V.mass_matrix() # Return the mass matrix scaled by the domain factor
        else:
            return float(h)**(1-num_derivatives) * assemble_generic_matrix(u, v) # Assemble and return the matrix
    return V.inner_product(u)           # Otherwise, compute the inner product directly

# Function to project a function ue onto a function space V
def project(ue, V):
    u = TrialFunction(V)                # Define the trial function
    v = TestFunction(V)                 # Define the test function
    b = inner(ue, v)                    # Compute the right-hand side
    A = inner(u, v)                     # Compute the matrix A
    uh = sparse.linalg.spsolve(A, b)    # Solve the linear system
    return uh                           # Return the solution coefficients

# Function to compute the L2 error between the numerical solution and the exact solution
def L2_error(uh, ue, V, kind='norm'):
    d = V.domain                        # Get the domain
    uej = sp.lambdify(x, ue)            # Convert the exact solution to a function
    def uv(xj): return (uej(xj)-V.eval(uh, xj))**2 # Define the error integrand
    return np.sqrt(quad(uv, float(d[0]), float(d[1]))[0]) # Perform integration and return the square root

# Test function for projection
def test_project():
    ue = sp.besselj(0, x)               # Define the exact solution using Bessel function
    domain = (0, 10)                    # Define the domain
    for space in (Chebyshev, Legendre): # Loop over the function spaces
        V = space(16, domain=domain)    # Create the function space
        u = project(ue, V)              # Project the exact solution onto the function space
        err = L2_error(u, ue, V)        # Compute the L2 error
        print(
            f'test_project: L2 error = {err:2.4e}, N = {V.N}, {V.__class__.__name__}') # Print the error
        assert err < 1e-6               # Assert that the error is small

# Test function for Helmholtz equation
def test_helmholtz():
    ue = sp.besselj(0, x)               # Define the exact solution using Bessel function
    f = ue.diff(x, 2)+ue                # Compute the right-hand side function f
    domain = (0, 10)                    # Define the domain
    for space in (NeumannChebyshev, NeumannLegendre, DirichletChebyshev, DirichletLegendre, Sines, Cosines): # Loop over function spaces
        if space in (NeumannChebyshev, NeumannLegendre, Cosines): # Check for Neumann or Cosines space
            bc = ue.diff(x, 1).subs(x, domain[0]), ue.diff(
                x, 1).subs(x, domain[1]) # Define Neumann boundary conditions
        else:
            bc = ue.subs(x, domain[0]), ue.subs(x, domain[1]) # Define Dirichlet boundary conditions
        N = 60 if space in (Sines, Cosines) else 12 # Set the number of basis functions
        V = space(N, domain=domain, bc=bc)         # Create the function space
        u = TrialFunction(V)                       # Define the trial function
        v = TestFunction(V)                        # Define the test function
        A = inner(u.diff(2), v) + inner(u, v)      # Assemble the matrix A for Helmholtz equation
        b = inner(f-(V.B.x.diff(x, 2)+V.B.x), v)   # Compute the right-hand side
        u_tilde = np.linalg.solve(A, b)            # Solve the linear system
        err = L2_error(u_tilde, ue, V)             # Compute the L2 error
        print(
            f'test_helmholtz: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}') # Print the error
        assert err < 1e-3                          # Assert that the error is small

# Test function for convection-diffusion equation
def test_convection_diffusion():
    eps = 0.05                         # Define small parameter for the problem
    ue = (sp.exp(-x/eps)-1)/(sp.exp(-1/eps)-1) # Define the exact solution
    f = 0                              # Define the right-hand side function f
    domain = (0, 1)                    # Define the domain
    for space in (DirichletLegendre, DirichletChebyshev, Sines): # Loop over function spaces
        N = 50 if space is Sines else 16 # Set the number of basis functions
        V = space(N, domain=domain, bc=(0, 1)) # Create the function space        u = TrialFunction(V)                   # Define the trial function
        v = TestFunction(V)                    # Define the test function
        A = inner(u.diff(2), v) + (1/eps)*inner(u.diff(1), v) # Assemble matrix A for convection-diffusion equation
        b = inner(f-((1/eps)*V.B.x.diff(x, 1)), v) # Compute the right-hand side
        u_tilde = np.linalg.solve(A, b)             # Solve the linear system
        err = L2_error(u_tilde, ue, V)              # Compute the L2 error
        print(
            f'test_convection_diffusion: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}') # Print the error
        assert err < 1e-3                           # Assert that the error is small

# Main block to execute test functions
if __name__ == '__main__':
    test_project()                   # Test projection
    test_convection_diffusion()      # Test convection-diffusion equation
    test_helmholtz()                 # Test Helmholtz equation

