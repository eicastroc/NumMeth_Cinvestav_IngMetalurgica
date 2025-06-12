import numpy as np

def euler(f, y0, x_span, h):
    """
    Implementación básica del método de Euler para resolver
    una ecuación diferencial ordinaria.
    
    Parameters
    ----------

    f : function
        Python function return a number. `f` must e continuous.
    y0: scalar
        initial condition f(0)= y0
    x_span: tupple
        two-value tupple with the initial and final xvalues, (x0,xf)
        for evaluation of the function.
    h : scalar
        step-size in the x-direction

    Returns
    -------
    X : array
        x-values where the differential equation is evaluated.
    Y : array
        y-values where the differential equation is evaluated.
    """
    x0, xf = x_span
    # preparar vectores X, Y con valores a evaluar
    X = np.arange(x0, xf+h, h)
    Y = np.empty_like(X)  
    Y[0] = y0

    # Método de euler (implementación explícita)
    for i, xi in enumerate(X[:-1]):
        Y[i+1] = Y[i] + f(X[i], Y[i])*h

    return X, Y


def rungeKutta4(f, y0, x_span, h):
    """
    Implementación básica del método de Runge-Kutta 4to orden 
    para resolver una ecuación diferencial ordinaria.
    
    Parameters
    ----------

    f : function
        Python function return a number. `f` must e continuous.
    y0: scalar
        initial condition f(0)= y0
    x_span: tupple
        two-value tupple with the initial and final xvalues, (x0,xf)
        for evaluation of the function.
    h : scalar
        step-size in the x-direction

    Returns
    -------
    X : array
        x-values where the differential equation is evaluated.
    Y : array
        y-values where the differential equation is evaluated.
    """
    x0, xf = x_span
    # preparar vectores X, Y con valores a evaluar
    X = np.arange(x0, xf+h, h)
    Y = np.empty_like(X)  
    Y[0] = y0

    # Método de Runge-Kutta 4 (implementación explícita)
    for i, xi in enumerate(X[:-1]):
        k1 = f(X[i], Y[i])
        k2 = f(X[i]+0.5*h, Y[i]+0.5*k1*h)
        k3 = f(X[i]+0.5*h, Y[i]+0.5*k2*h)
        k4 = f(X[i]+h, Y[i]+k3*h)
        Y[i+1] = Y[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)*h

    return X, Y