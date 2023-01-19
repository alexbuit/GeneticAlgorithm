######
Helper
######

This file contains the helper functions used all over the library. It is
not intended to be used directly by the user but can be useful in certain
situations.

floating point conversion routines
*********************************
The floating point to binary (and back) conversions support
the IEEE 754 standard and custom method proposed in :ref:`here <Binary representation
>`. Both these functions are intended for n > 2 matrices of binary values but
support single array conversions.

Plotting routines
*****************
There is a function to plot 3d functions (like the ones provided in
:ref:`Tests functions <Tests functions>`), this function is called
:func:`plot3d` and supports all plotting routines for 3d plots provided by
matplotlib.

Various functions
*****************

Sigmoid
-------
The file also contains a function for the sigmoid function in its more
simple form and a version containing all the different variables used in
the sigmoid function. A derivative of the sigmoid function is also provided.

Decorator test
--------------
The is_decorated function test if a function is decorated with a decorator.

Convertpop2bin
--------------
Converts a population of binary values to a population of integers.


Helper functions:
------------------
.. automodule:: dfmcontrol.helper
   :members:
   :undoc-members:
   :show-inheritance: