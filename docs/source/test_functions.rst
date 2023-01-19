Tests functions
===============

The t_functions file contains various mathematical functions that are used
to test the convergence of the genetic algorithm.  The functions are
described in the following table:


.. list-table::Title
   :widths: 25 25 50
   :header-rows: 1
    * - Function Name,
        - Function Type,
        - Description
    * - tfx
        - Unimodal
        - A simple unimodal function
    * - wheelers_ridge
        - Unimodal
        - A simple unimodal function
    * - booths_function
        - Unimodal
        - A simple unimodal function
    * - michealewicz
        - Unimodal
        - A simple unimodal function
    * - ackley
        - Unimodal
        - A simple unimodal function
    * - Styblinski_Tang
        - Unimodal
        - A simple unimodal function


All test functions included in this library are decorated with the
@tfxdecorator.  This decorator is used to provide the function with
the following attributes:
 - minimum: the minimum value of the function
 - maximum: the maximum value of the function (if available)
 - gradient: the gradient of the function
 - dim : the dimension of the function

The test functions can be found :ref:`here <Mathematical test functions>`.

.. toctree::
    Mathmatical test functions <t_functions>
    Minima lookup table <minima>