
A lookup table for minima
#########################

The following functions act as a lookup table for the minima of the
mathamtical functions described in :ref:`here <Mathematical test functions>`.
A minima for dimension n can be accesesed by calling the function with the
dimension as the only argument.

.. code-block:: python

    test_functions.minima.minStyblinski_Tang(2) # returns the minimal value for n = 2
    test_functions.minima.minStyblinski_Tangloc(2) # returns the location of the minimum

The min'func_name'loc functions returns the
location of the minima, while the min'func_name' functions returns the
function value at the minima.

.. note::
    Some functions are only valid for a certain
    amount of dimensions for these functions the parameter n is ignored.

Complete list of functions
--------------------------

All functions defined in this module are listed below.

.. automodule:: dfmcontrol.Mathematical_functions.minima
   :members:
   :undoc-members:
   :show-inheritance: