import numpy as np

def Lagrang(x, X, Y, n=None):
    """
    Basic implementation of Lagrange interpolation
    for a single prediction of nth degree,
    where n+1 is the number of data points

    Parameters
    ----------
    x: scalar
        x-value for which interpolation is sough.
    X: array
        x-values datapoints.
    Y: array
        y-values datapoints.
    n: int
        order of prediction, at most the number of datapoints - 1.

    Returns
    -------
    sum: scalar
        interpolated value of f(x).
    
    """

    if len(X) != len(Y):
        raise Exception(f"Data sets X,Y are not of the same length")

    if n == None:
        n = len(X)
    else:
        n = n+1 #necessry because of n exclusive counting in python

    if n > len(X):
        raise Exception(f"order of Lagrange polynomial higher than allowed")



    sum = 0
    for i in range(n):
        product = Y[i]
        for j in range(n):
            if i != j: product *= (x - X[j]) / (X[i] - X[j])
        sum += product
    return sum