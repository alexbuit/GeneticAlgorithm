#########
Selection        return x ** 2 + 1

#########

The selection of parents for the next generation is a vital part of the genetic algorithm
this library provides the following selection methods to do such:
 - Roulette Wheel Selection
 - Tournament Selection
 - Rank Selection
 - Boltzmann Selection

The methods are implemented by functions generally structured as the following

.. code-block:: python

    def selection_method(fx, population, probabiity, fitness, **kwargs):
        # Calcuate the fitness value by calling the fitness function
        y = fx(population)
        f = fitness(y)

        # Determine the probability of selection for each individual
        p = probability(f, **kwargs)

        # Select the individuals
        selected = np.random.choice(population, size=len(population), p=p)

        return selected, f, p, y

.. Note::
    The selection method needs to return 4 values:
        - The selected individuals
        - The fitness values of the selected individuals
        - The probability of selection of the selected individuals
        - The fitness values of the population
    for which the first one is important to the functioning of the genetic algorithm, the other
    three are for the user to use for debugging/logging purposes.

Roulette Wheel Selection
------------------------

The roulette wheel selection method is the most common selection method used in genetic
algorithms. It is based on the idea that the probability of selection of an individual is
proportional to its fitness value. The fitness values are normalized to a range of [0, 1]
and the individuals are selected based on the probability of selection of each individual.

So that the mathmatical formula for the probability of selection of an individual is


Tournament Selection
--------------------

Rank Selection
--------------

Boltzmann Selection
-------------------


Fitness evaluation
******************



Selection methods
-----------------

.. automodule:: dfmcontrol.selection_funcs
   :members:
   :undoc-members:
   :show-inheritance: