This folder contains test files for each py file,
these test simply check if the output matches up with an expected
output. The expected output is determinded as expected from the documentation.

Each file contains a check to see if each function in the py file has a corresponding
test function in the test file. If a function is not tested, the test will fail.

Population tests
----------------

The population tests are run by calling the function `test_population` in the
`test_population.py` file. This function will run the tests for the population
initialisation methods. Since these functions all try to achieve the same goal they
are tested by the same standards. The tests are as follows:
- The output is a numpy array
- The output has the correct shape, corresponding to the input
- The output is a 2D array

For the shape four tests are run:
- One individual, ten genes [1, 10]
- Ten individuals, one gene [10, 1]
- Ten individuals, ten genes [10, 10]
- One thousand individuals, one thousand genes [1000, 1000]

All other parameters are set to their default values. Which are:
- Bitsize of 32
- Bias of 0 
- Factor of 1 
- high of 1 
- low of 0 
- loc of 0 
- scale of 1 
- mu of 1 

The only exception to these tests is the `_create_pop` method which has some additional
tests. These tests are:
- A test for a cauchy distribution
- A test for a uniform distribution
- A test for a normal distribution