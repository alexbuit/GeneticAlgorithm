
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "Helper.h"

void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                float factor, float bias, int normalised, float** result){
    /*
    valarr is the array of bitarrays to be converted to integers (a x b)
    bitsize is the size of the bitarrays
    genes is the number of genes in the bitarrays (n = genes / bitsize; n = b / bitsize)
    individuals is the number of individuals in the bitarrays (m = individuals; m = a)
    result is the array of integers to be filled with the converted values (m x n)

    If normalised is 1, the values will be normalised to the range [0, 1] multiplied by factor and added bias
    If normalised is 0, the values will be normalised to the range [0, 2^bitsize - 1]
    */

    // check if the dimensions are correct
    if (sizeof(valarr) != sizeof(result)){
        printf("Error: valarr and result have different dimensions\n");
        exit(1);
    }

    // 

}

void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result){

    // check if the dimensions are correct
    if (sizeof(valmat) != sizeof(result)){
        printf("Error: valmat and result have different dimensions\n");
        exit(1);
    }

    // define the conversion matrix
    int** a = malloc(bitsize * sizeof(int*)); 
    for (int i = 0; i < bitsize; i++){
        a[i] = malloc(bitsize * sizeof(int));
    }

    for (int i = 0; i < bitsize; i++){
        for (int j = 0; j < bitsize; j++){
            a[i][j] = pow(2, j);
        }
    }

    // convert the values with valmat @ a
    for (int i = 0; i < individuals; i++){
        for (int j = 0; j < genes; j++){
            for (int k = 0; k < bitsize; k++){
                result[i][j] += valmat[i][k] * a[k][j];
            }
        }
    }
}