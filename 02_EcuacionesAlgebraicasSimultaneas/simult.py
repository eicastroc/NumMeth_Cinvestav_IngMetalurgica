import numpy as np


def gaussSeidel(A, b, x, relax=1.0, tol=None, niter=100):
    """
    Basic Gauss-Seidel algorithm

    Parameters
    ----------
    A : array
        LHS term, matrix of coefficients of the system of equations
    b : array
        RHS term, vector of coefficients of the system of equations
    x : array
        initial guesses for the unknowns, x.
    relax : scalar
        relaxation factor, a scalar between 0 and 1.
    tol : float
        Tolerance of the numerical method. Must be a positive number.
        Default is None, which means that convergence is not evaluated.
    niter : float
        maximum number of iterations of the method.
        Default value is 100.

    Returns
    -------
    x : float
        value of x (array of unknowns) at the last iteration.
    n: float
        number of iterations of the method
    
    """
    
    if relax < 0.0 or relax > 2.0:
        raise Exception(f"relax value is {relax}, it should be between 0.0 and 2.0")

    if A.shape[0] != A.shape[1]:
        raise Exception(f"The matrix A is not a square matrix, it has {A.shape[0]} rows and {A.shape[1]} columns")
    
    if len(A) != len(b) != len(x):
        raise Exception("The size of vectors b or x are different than the number of rows in matrix A")
    
    n = A.shape[0] # number of rows in matrix 
    iter = 0
    for iter in range(1,niter+1):
        x_old = x.copy()
        # bucle sobre las filas
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j!=i:
                    sigma += A[i,j] * x[j]
            x[i] = (b[i] - sigma) / A[i,i]
            
        x = relax*x + (1-relax) * x_old
        err = np.abs((x-x_old)/x)

        if tol!=None and max(err) < tol:
                print(f"Convergence has been reached after {iter} iterations")
                break

    return x, iter