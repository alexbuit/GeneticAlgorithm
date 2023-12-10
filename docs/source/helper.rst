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

When using C
************

This file is also included in the C library and is used to convert the 
integers to binary values and in reverse. The functions are called:

integer to binary
- :func:`int2bin`
- :func:`intarr2binarr`
- :func:`intmat2binmat`

binary to integer
- :func:`bin2int`
- :func:`binarr2intarr`
- :func:`binmat2intmat`

Furthermore, the more high level bin to float and float to bin functions are
also included in the C library. These functions are named like the python
functions and can both be used for integers and floats.

These functions include:
- :func:`ndbit2int`
- :func:`ndint2bit`

And use the same bias and factor parameters as the python functions to compute
the floating point values.

note:: The C library does not initialise arrays/matrices these should be
       initialised before calling the functions in the form of a value-array,
       the amount of genes/individuals/bitsize and a correctly sized result-array.
       The functions will not check if the result-array is correctly sized 
       and will overwrite any values in the result-array.

The C library also includes the sigmoid function and its derivative. These
functions are named as in the python library. The sigmoid functions operate
on arrays and are used in the same way as the conversion functions where it
is expected that the result-array is correctly sized.


Helper functions:
------------------
.. automodule:: dfmcontrol.Helper.helper
   :members:
   :undoc-members:
   :show-inheritance: