#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.c"
#include "pop.h"

void bitpop(int bitsize, int genes, int individuals, int** result){

    /*
    Fill a matrix with random bits.

    Parameters
    ----------
    bitsize : int
        The size of the bitstring.

    genes : int
        The number of genes in the bitstring.

    individuals : int
        The number of individuals in the bitstring.

    result : int**
        The matrix to be filled with random bits.
        shape = (individuals, genes * bitsize)

    */

   for(int i=0; i<individuals; i++){
       for(int j=0; j<genes * bitsize; j++){
           result[i][j] = rand() % 2;
       }
   }

}
void uniform_bit_pop(int bitsize, int genes, int individuals,
                     float factor, float bias, int normalised, int** result){
    // printf("uniform_bit_pop\n");

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
