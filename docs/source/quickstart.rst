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
:math:`f(x) = 3 x^2 + 2x + 1`. With its minimum at :math:`x = -\dfrac{1}{3}`.

2.1 Import the necessary modules
--------------------------------

.. code-block:: python

    import numpy as np
    from dfmcontrol import GeneticAlgorithm

