
import numpy as np

"""
Gradient descent algortihm for use with the genetic algorithm
"""

def gradient_descend(f, x0, args=(), alpha=0.01, max_iter=50, tol=1e-6, verbose=False):
    """
    Gradient descent algorithm for use with the genetic algorithm

    :param f: function to be minimized
    :param x0: starting point
    :param args: arguments for f
    :param alpha: step size
    :param max_iter: maximum number of iterations
    :param tol: tolerance for convergence
    :param verbose: print information

    :return: x, f(x)
    """
    x = x0
    for i in range(max_iter):
        x_new = x - alpha * f(x, *args)
        if np.linalg.norm(x_new - x) < tol:
            if verbose:
                print("Converged after {} iterations".format(i))
            break
        x = x_new
    return x, f(x, *args)

if __name__ == "__main__":
    def f(x):
        return x**2 + x

    x, fx = gradient_descend(f, 4, alpha=0.001, max_iter=1000)
    print("x: {}, f(x): {}".format(x, fx))
