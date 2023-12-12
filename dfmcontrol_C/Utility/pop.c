#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.c"
#include "pop.h"

void bitpop(int bitsize, int genes, int individuals, int** result){

    /*
    Fill a matrix with random bits.

    :param bitsize: The size of the integer in binary.
    :type bitsize: int

    :param genes: The number of genes in an individual.
    :type genes: int

    :param individuals: The number of individuals in the population.
    :type individuals: int

    :param result: The matrix to be filled with random bits.
                   shape = (individuals, genes * bitsize)
    :type result: int**

    */

   for(int i=0; i<individuals; i++){
       for(int j=0; j<genes * bitsize; j++){
           result[i][j] = rand() % 2;
       }
   }

}
void uniform_bit_pop(int bitsize, int genes, int individuals,
                     float factor, float bias, int normalised, int** result){
    /*
    Fill a matrix with bits according to a uniform distribution.

    :param bitsize: The size of the bitstring.
    :type bitsize: int

    :param genes: The number of genes in the bitstring.
    :type genes: int

    :param individuals: The number of individuals in the bitstring.
    :type individuals: int

    :param factor: The factor by which the uniform distribution is scaled.
    :type factor: float

    :param bias: The bias of the uniform distribution.
    :type bias: float

    The factor and bias are used to calculate the upper and lower bounds of the uniform distribution 
    according to the following formula: [1]

    .. math::
        upper = round((bias + factor) * 2^{bitsize}) \\
        lower = round((bias - factor) * 2^{bitsize})

    Which results in the integer domain between round(lower * 2^{bitsize}) and round(upper * 2^{bitsize}).

    :param normalised: Whether the uniform distribution is normalised.
    :type normalised: int


    :param result: The matrix to be filled with bits according to a uniform distribution.
                   shape = (individuals, genes * bitsize)
    :type result: int**

    References
    ----------
    .. [1] https://stackoverflow.com/questions/11641629/generating-a-uniform-distribution-of-integers-in-c Lior Kogan (2012)

    */

    int upper = (int) roundf((bias + factor) * pow(2, bitsize));
    int lower = (int) roundf((bias - factor) * pow(2, bitsize));

    int** temp = malloc(sizeof(int) * individuals );

    for(int i=0; i<individuals; i++){
        temp[i] = (int*) malloc(sizeof(int) * genes * bitsize);
    }

    // now we have upper and lower bounds for the uniform distribution in the 
    // integer domain between round(lower * 2^bitsize) and round(upper * 2^bitsize)

    unsigned int nRange = (unsigned int)(upper - lower);
    unsigned int nRangeBits = (unsigned int) ceil(log2((double) (nRange)));

    // now we have the number of bits required to represent the range of the
    // uniform distribution in the integer domain

    unsigned int nRand; // random number

    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes * bitsize; j++){
            do{
                nRand = 0;
                for(int k=0; k<nRangeBits; k++){
                    nRand = (nRand << 1) | (rand() & 1);
                }
            } while(nRand >= nRange);
            temp[i][j] = (int) (nRand + lower);
        }
    }

    // now we have a matrix of random integers between round(lower * 2^bitsize)
    // and round(upper * 2^bitsize) which we can convert to a matrix of bits

    intmat2binmat( temp, bitsize, genes, individuals, result);

    // free the memory
    for(int i=0; i<individuals; i++){
        free(temp[i]);
    }

    free(temp);
}
void normal_bit_pop(int bitsize, int genes, int individuals,
                    float factor, float bias, int normalised,
                    float loc, float scale, int** result){
    /*
    
    */

}
void cauchy_bit_pop(int bitsize, int genes, int individuals,
                    float factor, float bias, int normalised,
                    float loc, float scale, int** result){
    /*
    
    */

}
