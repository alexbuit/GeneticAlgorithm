########
Examples
########

Optimizing a 1d test function
#############################

.. code-block:: python

    from dfmcontrol.DFM_opt_alg import genetic_algoritm
    import dfmcontrol as dfmc
    from dfmcontrol.test_functions import tfx

    import dfmcontrol.helper as dfmh

    # Create a genetic algorithm object
    ga = genetic_algoritm(bitsize=16)
    tfunc = tfx

    # use the default bit2num function
    ga.b2n = dfmh.ndbit2int

    # The range of possible floats is now -10 to 10
    ga.b2nkwargs = {"factor": 10}

    # Initiate a population of 16 individuals with 2 genes each
    ga.init_pop("normal", shape=[20, 1], bitsize=ga.bitsize, factor=ga.b2nkwargs["factor"])

    # Initiate the log
    ga.logdata(2)

    # Set the target function
    ga.target_func(tfunc)

    # Set the selection, mutation and crossover functions
    ga.set_select(dfmc.selection_funcs.rank_tournament_selection) # Rank selection
    ga.set_mutate(dfmc.mutation.full_mutate) # Mutate the full bit (do not use for IEEE 754 floats)
    ga.set_cross(dfmc.cross_funcs.full_equal_prob) # Crossover the full bit (do not use for IEEE 754 floats)

    ga.elitism = 4  # Keep the 4 best individuals
    # Run the genetic algorithm

    # The runcond argument is a string that is evaluated as a condition for the algorithm to stop
    # The string can use any of the attributes of the genetic algorithm object
    runcond = r"np.min(np.abs(self.log.ranking.distancefx[-1])) > 0.01"  # Stop when the minimum distance from the best solution to the mathematical is less than 0.1

    # The verbosity argument is an integer that controls the amount of output
    # 0: No output
    # 1: Output the current generation, (best & group) distance to the best solution and the best fitness
    verbosity = 2

    # The muargs argument is a dictionary that is passed to the mutation function as the kwargs argument
    muargs = {"mutate_coeff": 4}

    # The selargs argument is a dictionary that is passed to the selection function as the kwargs argument
    selargs = {"nbit2num": ga.b2n, "fitness_func": dfmc.selection_funcs.no_fitness,
                "allow_duplicates": True}

    ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity, runcond=runcond)

    # Save the results to a txt file
    ga.log.ranking.save2txt("result", "results.txt")

Plotting the results
====================

Optimizing Ackley's function for 2 variables
############################################

Ackley's function is defined in :ref:`test functions <Ackley>`. The following code
finds the minimum of the function using the genetic algorithm.

.. code-block:: python

    from dfmcontrol.DFM_opt_alg import genetic_algoritm
    import dfmcontrol as dfmc
    from dfmcontrol.test_functions import ackley

    import dfmcontrol.helper as dfmh

    # Create a genetic algorithm object
    ga = genetic_algoritm(bitsize=16)
    tfunc = ackley

    # use the default bit2num function
    ga.b2n = dfmh.ndbit2int

    # The range of possible floats is now -5 to 5
    ga.b2nkwargs = {"factor": 20}

    # Initiate a population of 16 individuals with 2 genes each
    ga.init_pop("normal", shape=[40, 2], bitsize=ga.bitsize, factor=ga.b2nkwargs["factor"])

    # Initiate the log
    ga.logdata(2)

    # Set the target function
    ga.target_func(tfunc)

    # Set the selection, mutation and crossover functions
    ga.set_select(dfmc.selection_funcs.rank_tournament_selection) # Rank selection
    ga.set_mutate(dfmc.mutation.full_mutate) # Mutate the full bit (do not use for IEEE 754 floats)
    ga.set_cross(dfmc.cross_funcs.full_equal_prob) # Crossover the full bit (do not use for IEEE 754 floats)

    ga.elitism = 4  # Keep the 4 best individuals
    # Run the genetic algorithm

    # The runcond argument is a string that is evaluated as a condition for the algorithm to stop
    # The string can use any of the attributes of the genetic algorithm object
    runcond = r"np.min(np.abs(self.log.ranking.distancefx[-1])) > 0.1"  # Stop when the minimum distance from the best solution to the mathematical is less than 0.1

    # The verbosity argument is an integer that controls the amount of output
    # 0: No output
    # 1: Output the current generation, (best & group) distance to the best solution and the best fitness
    verbosity = 1

    # The muargs argument is a dictionary that is passed to the mutation function as the kwargs argument
    muargs = {"mutate_coeff": 2}

    # The selargs argument is a dictionary that is passed to the selection function as the kwargs argument
    selargs = {"nbit2num": ga.b2n, "fitness_func": dfmc.selection_funcs.no_fitness,
                "allow_duplicates": True}

    for i in range(1):
        ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity, runcond=runcond)
        ga.reset(False)

    # Save the log object to a .pickle file to be able to retrieve results later.
    ga.save_log("log2.pickle")

Acquiring the results from a saved log
======================================

.. code-block:: python

    import numpy as np
    from dfmcontrol.DFM_opt_alg import genetic_algoritm

    import matplotlib.pyplot as plt

    ga = genetic_algoritm(bitsize=16)
    ga.load_log("log2.pickle")

    log = ga.log

    plt.plot(np.arange(len(log.ranking.result)), [np.average(i) for i in log.ranking.result], label="Group average")
    plt.plot(np.arange(len(log.ranking.distancefx)), [np.min(i) for i in log.ranking.distancefx], label="Best result")

    plt.xlabel("Generation")
    plt.ylabel("Distance to the best solution")

    plt.title("Result of the genetic algorithm")

    plt.legend()

    plt.show()

Which results in the following plot:

.. figure:: /_images/2d_ackley.png

.. note::
    Due to the stochastic nature of the genetic algorithm, the results will vary from run to run.

Optimizing the 39 dimensional Styblinski-Tang function
#######################################################

The Styblinski-Tang function is defined in :ref:`test functions <Styblinski-Tang>`. The following code
finds the minimum of the function using the genetic algorithm.

.. code-block:: python

    from dfmcontrol.DFM_opt_alg import genetic_algoritm
    import dfmcontrol as dfmc
    from dfmcontrol.test_functions import Styblinski_Tang

    import dfmcontrol.helper as dfmh

    # Create a genetic algorithm object
    ga = genetic_algoritm(bitsize=16)
    tfunc = Styblinski_Tang

    # use the default bit2num function
    ga.b2n = dfmh.ndbit2int

    # The range of possible floats is now -5 to 5
    ga.b2nkwargs = {"factor": 5}

    # Initiate a population of 40 individuals with 39 genes each
    ga.init_pop("normal", shape=[40, 39], bitsize=ga.bitsize, factor=ga.b2nkwargs["factor"])

    # Initiate the log
    ga.logdata(2)

    # Set the target function
    ga.target_func(tfunc)

    # Set the selection, mutation and crossover functions
    ga.set_select(dfmc.selection_funcs.rank_tournament_selection) # Rank selection
    ga.set_mutate(dfmc.mutation.full_mutate) # Mutate the full bit (do not use for IEEE 754 floats)
    ga.set_cross(dfmc.cross_funcs.full_equal_prob) # Crossover the full bit (do not use for IEEE 754 floats)

    ga.elitism = 10  # Keep the 10 best individuals
    # Run the genetic algorithm

    # The runcond argument is a string that is evaluated as a condition for the algorithm to stop
    # The string can use any of the attributes of the genetic algorithm object
    runcond = r"np.min(np.abs(self.log.ranking.distancefx[-1])) > 0.1"  # Stop when the minimum distance from the best solution to the mathematical is less than 0.1

    # The verbosity argument is an integer that controls the amount of output
    # 0: No output
    # 1: Output the current generation, (best & group) distance to the best solution and the best fitness
    verbosity = 1

    # The muargs argument is a dictionary that is passed to the mutation function as the kwargs argument
    muargs = {"mutate_coeff": 3}

    # The selargs argument is a dictionary that is passed to the selection function as the kwargs argument
    selargs = {"nbit2num": ga.b2n, "fitness_func": dfmc.selection_funcs.no_fitness,
                "allow_duplicates": True}

    ga.run(muargs=muargs, selargs=selargs, verbosity=verbosity, runcond=runcond)

    # Save the log object to a .pickle file to be able to retrieve results later.
    ga.save_log("log3.pickle")

The results are shown in the following plot:

.. figure:: /_images/39d_styb.png

The log object can also be used to extract data on the time / calculations required to find the minimum.

.. code-block:: python

    from dfmcontrol.AdrianPackv402.Aplot import Default

    pl = Default(log.time.data, log.time.calculation, x_label="Time", y_label="Requests to the test function", degree=1,
                 marker="")
    pl()

.. figure:: /_images/39_styb_calculation.png