import numpy as np
import matplotlib.pyplot as plt


def trapzf(f, A, B, N):
    xx = np.linspace(A, B, N+1)
    integral = 0
    for idx, x, in enumerate(xx[:-1]):
        a, b = xx[idx], xx[idx+1]
        integral += (b - a) / 2 * (f(a) + f(b))
    return integral


def trapzd(xx, yy):
    if len(xx) != len(yy):
        raise Exception("Las series de datos x, y no tienen el mismo tamaño")
    integral = 0
    for idx, x, in enumerate(xx[:-1]):
        a, b = xx[idx], xx[idx+1]
        fa, fb = yy[idx], yy[idx+1]
        integral += (b - a) / 2 * (fa + fb)
    return integral 


def simpsf(f, A, B, N):
    xx = np.linspace(A, B, N+1)
    integral = 0
    for idx, x, in enumerate(xx[:-1]):
        a, b = xx[idx], xx[idx+1]
        mp = (a + b) / 2
        integral += (b - a) / 6 * (f(a) + 4*f(mp) + f(b))
    return integral



def plot_numerical_integration(f, A, B, N, method='midpoint', ax=None):
    """
    Taken from: https://link.springer.com/book/10.1007/978-1-4842-4246-9
    
    """

    if ax == None:
        fig, ax = plt.subfigures()
    valid_methods = ['midpoint', 'trapezoid', 'simpson']
    if method not in valid_methods:
        raise ValueError("valid methods are: {valid_methods}")
        
    # plot the "true" area
    xx = np.linspace(A, B, 100)
    ax.fill_between(xx, f(xx), color='lightgray', alpha=0.25)
    
    # divide domain into the N-composite quadrature sub-intervals
    xx = np.linspace(A, B, N+1)
    ax.plot(xx, f(xx), 'ro')
    
    # Do the numerical integration
    integral = 0
    for idx, x in enumerate(xx[:-1]):
        a = xx[idx]
        b = xx[idx+1]
        if method=='midpoint':
            # integrate for a,b sub-interval
            mp = (a + b) / 2
            integral += (b - a) * f(mp)
            # plot the approximation "polygons"
            ax.plot(mp, f(mp), ls='', marker='x', color='k')
            ax.stem([a, b], [f(mp), f(mp)], basefmt='k:', linefmt='k:', markerfmt='None')
            ax.plot([a, b], [f(mp), f(mp)], 'k:')
        elif method=='trapezoid':
            # integrate for a,b sub-interval
            integral += (b - a) / 2 * (f(a) + f(b))
            # plot the approximation "polygons"
            ax.stem([a, b], [f(a), f(b)], basefmt='k:', linefmt='k:', markerfmt='None')
            ax.plot([a, b], [f(a), f(b)], 'k:')
        elif method=='simpson':
            # integrate for a,b sub-interval
            mp = (a + b) / 2
            integral += (b - a) / 6 * (f(a) + 4*f(mp) + f(b))
            # plot the approximation "polygons"
            ax.stem([a, b], [f(a), f(b)], basefmt='k:', linefmt='k:', markerfmt='None')
            f2 = np.polynomial.Polynomial.fit([a, mp, b], [f(a), f(mp), f(b)], 2)
            x_array = np.linspace(a, b, 25)
            ax.plot(x_array, f2(x_array), 'k:')
            

    # Annotate the plot with the value of the numerical integration
    ax.annotate(r"$\int_a^b f(x) dx \approx {:.6f}$".format(integral),
               xy=(0.1, 0.7), xycoords='axes fraction')
    