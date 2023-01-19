###########################
Mathematical test functions
###########################

This library contains a number of mathematical test functions to test the
convergence of optimization algorithms. The functions are taken from the
following sources:
 - simone fraser university :ref:`http://www.sfu.ca/~ssurjano/optimization.html` [1]
 - Algorithms for Optimization by Mykel J. Kochenderfer & Tim A. Wheeler [2]
 - Certified global minima for a benchmark of difficult optimization problems [3]

The functions are implemented in the :mod:`test_functions` module.

.. [1] S. Fraser, "Test functions for optimization algorithms," 2013.
.. [2] M. J. Kochenderfer & T. A. Wheeler, "Algorithms for Optimization," 2013.
.. [3] "Certified global minima for a benchmark of difficult optimization problems," 2013.

Functions
#########

1-d functions
=============

tfx
---
.. autofunction:: tfx


2-d functions
=============

wheelers ridge
--------------
.. autofunction:: wheelers_ridge

booths function
---------------
.. autofunction:: booths_function

n-d functions
=============

Ackley
------






.. automodule:: dfmcontrol.test_functions.t_functions
   :members:
   :undoc-members:
   :show-inheritance:

    .. autodecorator:: dfmcontrol.test_functions.t_functions.ackley
