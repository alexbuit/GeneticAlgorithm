=================
Genetic Algorithm
=================

Summary:
========
The genetic algorothm provided by the library is a simple implementation of the
standard genetic algorithm. It is not optimized for speed, but it is easy to
understand and to use. The algorithm proposed by the library also has a lot
of options for customisation wherein functions used for:
 - population initialization
 - Selection
    - fitness evaluation
 - Crossover
 - Mutation

can be either user provideded or selected from the library. All the library
provided functions are documented on the wiki and can be found in the
following pages:
 - :ref:`Population <Population>`
 - :ref:`Selection <Selection>`
 - :ref:`Fitness evaluation <Fitness evaluation>`
 - :ref:`Crossover <Crossing>`
 - :ref:`Mutation <Mutation>`

The algorithm supports both maximization and minimization of the fitness
function. The type of optimization is specified by the user in the selection
function.

Binary representation
---------------------

The algorithm supports two types of binary representation of floating points.
 - The standardised IEEE 754 floating point representation.
 - A custom representation

The algorithm works best with the custom representation of a floating point,
this representation assigns an integer value to each floating point value. This
integer value is converted to a binary string and used as the chromosome.
The amount of floating point values that can be represented by the custom
representation is limited by the size of the integer used to represent the
floating point value and the upper and lower boundaries used in the conversion.
Typically a value is converted as follows:

.. code-block:: python

    def convert_to_binary(value: float, factor: float, bias: float, bitsize: int) -> np.ndarray:
        """
         Converts a floating point value to a binary array. Assuming the
            floating point value is in the range [-factor + bias, factor + bias]
            the binary array will have a length of bitsize.
        """
        # normalize the value
        value = (value - bias) / factor
        # convert to integer
        value = int(value * (2 ** bitsize - 1))

        # Using an int2binary converter for conversion
        return i2b(value, bitsize)

And to convert back:

.. code-block:: python

    def convert_from_binary(binary: np.ndarray, factor: float, bias: float) -> float:
        """
        Converts a binary array to a floating point value. Assuming the
        binary array has a length of bitsize, the floating point value will
        be in the range [-factor + bias, factor + bias]
        """
        # convert to integer
        value = b2i(binary)
        # convert to floating point
        value = value / (2 ** len(binary) - 1) * factor + bias
        return value

These functions are provided by the library and can be found in the
helper file, :ref:`Helper <Helper>`

General usage
=============

run config
----------

The algorithm is used by creating a :ref:`GeneticAlgorithm <GeneticAlgorithm>`
this instance only requires the bitsize but this value is intiansiated with
the default value of 32. After initialising a :ref:`GeneticAlgorithm <GeneticAlgorithm>`
instance the default functions can be replaced by the user provided functions
or other functions in the library. The functions that can be replaced are:
 - :ref:`Population <Population>`
 - :ref:`Selection <Selection>`
 - :ref:`Fitness evaluation <Fitness evaluation>`
 - :ref:`Crossover <Crossing>`
 - :ref:`Mutation <Mutation>`

A function can be set directly by using the following syntax:

.. code-block:: python
    ga.population = ga.population_random

or by a method provided by the class:

.. code-block:: python
    ga.set_population(ga.population_random)

.. Note::
    Setting a value directly will not check if the function is valid for the
    algorithm or adjust any other values within the algorithm that might need adjusting.

Running the algorithm
---------------------

Once the run settings have been set the algorithm can be run by calling the
class or by calling the :meth:`run <GeneticAlgorithm.run>` method.

.. code-block:: python

    ga()  # will do the same as ga.run()

Running the class often requires additional arguments to be passed to the
run method. These arguments are used as kwargs for the (user provided) functions
and are added seperatly as the following
 - selargs, for the selection function
 - cargs, for the crossover function
 - muargs, for the mutation function

As shown in the following example:
.. code-block:: python

    ga(selargs={'maximize': True}, cargs={'crossover_rate': 0.5}, muargs={'mutation_rate': 0.1})

.. Note::
    The arguments for the fitness function are often added as part of the selargs argument.

The verbosity of the algorithm can be adjusted by setting the verbosity level
of the algortihm to one of the following values:
 - 0: No output
 - 1: Output the best fitness value of each generation
 - 2: Output the best fitness value of each generation and the best chromosome

which can be done by using the following syntax:
.. code-block:: python

    ga.verbosity = 2

or by adding this as a kwarg to the run method:
.. code-block:: python

    ga.run(verbosity=2)

The maximum number of generations can be set by using the following syntax:
.. code-block:: python

    ga.run(epochs=100)

Saving and loading results
--------------------------

Results can be saved by using the :meth:`save <GeneticAlgorithm.savelog>` method,
this will save the

Methods
=======

.. automodule:: dfmcontrol.DFM_opt_alg
    :members:
    :undoc-members:
    :show-inheritance: