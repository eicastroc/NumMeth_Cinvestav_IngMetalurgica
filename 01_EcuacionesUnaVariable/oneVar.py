import numpy as np


def bisection(f, l, u, tol=None, niter=100):
    """
    Basic bisection algorithm for finding the root of a one-variable equation

    Parameters
    ----------
    f : function
        Python function returning a number.  `f` must be continuous, and
        f(l) and f(u) must have opposite signs.
    l : scalar
        lower bound for the bisection method
    u : scalar
        upper bound for the bisection method
    tol : float
        Tolerance of the numerical method. Must be a positive number.
        Default is None, which means that convergence is not evaluated.
    niter : float
        maximum number of iterations of the method.
        Default value is 100.

    Returns
    -------
    m : float
        value of xm (midpoint) at the last iteration.
    n: float
        number of iterations of the method
    
    """
    fl, fu = f(l), f(u)

    # 1. verificar que el intervalo inicial contenga por lo menos una raíz
    if np.sign(fl) == np.sign(fu):
        raise Exception("No hay raíz en el intervalo especificado, [{l} : {u}]")
    m = l + (u-l)/2
    fm = f(m)

    for n in range(1,niter+1):
        # 2. encontrar el punto medio del intervalo
        m = l + (u-l)/2
        fm = f(m)
        # 3. Determinar el nuevo intervalo donde se encuentra la raíz
        if np.sign(fl) != np.sign(fm):
            u, fu = m, fm
        else:
            l, fl = m, fm
            
        if tol!=None and abs(f(m)) < tol:
                #print(f"Convergence has been reached!")
                break

    return m, n



def fixedPoint(f, g, xi, lamb=1.0, tol=None, niter=10):
    """
    Basic bisection algorithm for finding the root of a one-variable equation

    Parameters
    ----------
    f : function
        Python function returning a number.  `f` must be continuous.
    g : function
        Python function returning a number. `g` must be continuous.
        g is the "root-predictor" function.
    xi : scalar
        initial guess for root.
    lamb : scalar
        relaxation factor
    tol : scalar
        tolerance of the method, [0, 1]
    niter : int
        maximum number of iterations of the method.
        Default value is 100.

    Returns
    -------
    m : float
        value of xm (midpoint) at the last iteration.
    n: float
        number of iterations of the method
    
    """
    for n in range(1,niter+1):
        x0 = xi
        xi = g(x0)
        xi = lamb*xi + (1-lamb)*x0

        if tol!=None and abs(f(xi))<tol:
                break
    return xi,n


def newton_raphson(f, fp, xi, lamb=1.0, tol=None, niter=10):
    """
    Basic bisection algorithm for finding the root of a one-variable equation

    Parameters
    ----------
    f : function
        Python function returning a number.  `f` must be continuous.
    fp : function
        Python function returning a number. `fp` must be continuous.
        `fp` is the derivative of `f`.
    xi : scalar
        initial guess for root.
    lamb : scalar
        relaxation factor
    tol : scalar
        tolerance of the method, [0, 1]
    niter : int
        maximum number of iterations of the method.
        Default value is 100.

    Returns
    -------
    m : float
        value of xm (midpoint) at the last iteration.
    n: float
        number of iterations of the method
    
    """
    for n in range(1,niter+1):
        x0 = xi
        xi = xi - f(xi)/fp(xi)
        xi = lamb*xi + (1-lamb)*x0

        if tol!=None and abs(f(xi))<tol:
                break
    return xi,n