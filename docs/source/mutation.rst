########
Mutation
########

The mutation of the population is an essential part of the evolutionary algorithm. The mutation is used to introduce new genetic material into the population. 
This is done by randomly changing the binary values of some of the genes. The mutation is done with a certain probability of flipping.


Mutation methods
---------------------------------
The main method used for mutation is the :func:`dfmcontrol.Utility.mutation.mutation` method. This method is available for
the IEEE binary coded GA and the project specific GA in python and C. 

The mutation coefficient is defined as the amount of mutations that are done in the population. The default value is one bit per gene, calculated by:

.. math::
   \text{mutation coefficient} = \frac{\text{number of genes}}{\text{number of individuals}}

The mutation coefficient can be changed by the user by adjusting the kwarg ``mutation_coefficient``. The mutation coefficient can be set to a value between 0 and bitsize * genes.

Methods in python
=================

.. automodule:: dfmcontrol.Utility.mutation
   :members:
   :undoc-members:
   :show-inheritance:


Methods in C
============

.. automodule:: dfmcontrol_C.Utility.mutation
   :members:
   :undoc-members:
   :show-inheritance: