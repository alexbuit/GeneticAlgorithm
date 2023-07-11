##########
Quickstart
##########

This guide will explain how to install and configure a genetic algorithm using
the library.

1. Install the package
=======================
The package can be installed with pip and git:

.. code-block:: bash

    pip install git+https://github.com/AdrianvEik/KTH_dfm_control#egg=dfmcontrol

2. Run a simple optimization algorithm.
=======================================

To show the basic functionality of the library, we will optimize a simple
function defined :ref:`here <tfx>` as :func:`tfx`. The function is defined as:
:math:`f(x) = (3 x^2 + 2x + 1) \cdot \sin{(x)}`. With its minimum at
:math:`x = 0`.



2.1 Import the necessary modules
--------------------------------

.. code-block:: python

    import numpy as np
    from dfmcontrol import GeneticAlgorithm
    from dfmcontrol.test_functions import tfx

2.2 Initialize the genetic algorithm
------------------------------------
The following code initializes the genetic algorithm with a bitsize of 16.
The bitsize is the number of bits used to represent the individual in the population.
This binary value is then converted to a numerical value using the defined
:attr:`b2n <dfmcontrol.genetic_algorithm.b2n>` with a function from
:ref:`Helper <Helper>`, :func:`ndbit2int`. The attribute :attr:`b2nkwargs
<dfmcontrol.genetic_algorithm.b2nkwargs>` defines the keyword arguments for the
conversion function, setting the lower and upper bound of the numerical value.

.. code-block:: python

    # We use a bit size of 16 to represent the values in the range [-10, 10]
    ga = genetic_algorithm(bitsize=16)

    # Initialize the conversion function for binary to numerical values
    ga.b2n = dfmh.ndbit2int
    ga.b2nkwargs = {"factor": 10} # This defines the search space [-10, 10]

.. Note::
    The conversion function can be any function that takes a binary array and
    returns a numerical value. The function must be defined in the module
    :mod:`dfmcontrol.helpers <dfmcontrol.helpers>`. This function will also
    define the population initailisation that is available in the module.

2.3 Initialise the population.
-------------------------------
The population is initialised using the :func:`init_pop` method of the
genetic algorithm. Which as the following arguments:

- Method: The method used to initialise the population. This can be either
  :attr:`uniform_bit_pop <dfmcontrol.pop.uniform_bit_pop>` or
  :attr:`cauchy_bit_pop <dfmcontrol.pop.cauchy_bit_pop>` or
  :attr:`bitpop <dfmcontrol.pop.bitpop>` from .