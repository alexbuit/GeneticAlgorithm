
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../../Helper/Helper.c"
// #include "../../Helper/Helper.h"

int main(){

    int individuals = 4;
    int genes = 4;
    int bitsize = 32;

    int factor = 10.4;
    int bias = 0;
    int normalised = 1;


    float** valmat = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of integers
    int** result = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of bitarrays
    float** copyvalmat = malloc(individuals * sizeof(int*)); // copy of valmat

    for(int i=0; i<individuals; i++){
        valmat[i] = malloc(genes * sizeof(float));
        result[i] = malloc(genes * bitsize * sizeof(int));
        copyvalmat[i] = malloc(genes * sizeof(float));
    }

    // fill valmat with random integers
    int sign;
    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            sign = (rand() % 2 == 0) ? 1 : -1;
            valmat[i][j] = (float) (rand() % 9 * sign); // random integer between -2^bitsize-1 and 2^bitsize - 1
            copyvalmat[i][j] = valmat[i][j];
        }
    }

    // cast the values to floats
    for (int i = 0; i < individuals; i++){
        for (int j = 0; j < genes; j++){
            valmat[i][j] = (float) valmat[i][j];
        }
    }

    // convert the values to bitarrays
    int2ndbit(valmat, bitsize, genes, individuals, factor, bias, result);

    printf("result: \n");
    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            printf("%#10x", result[i][j]);
            // for(uint32_t k = 1; k != 0x00000000; k = k << 1){
            //     printf("%c", (result[i][j]&k ? '1':'0'));
            // }
        }
        printf("\n");
    }

    // print the size of the matrices and the values
    printf("Size of valmat: %d x %d\n", individuals, genes);
    printf("Size of result: %d x %d\n", individuals, genes * bitsize);

    printf("valmat: \n");
    printfMatrix(copyvalmat, individuals, genes, 6);

    printf("\n");

    // Check the result by doing the reverse operation
    int** same = malloc(individuals * sizeof(int*));
    for(int i=0; i<individuals; i++){
        same[i] = malloc(genes * sizeof(int));
    }

    int same_count = 0;

    ndbit2int(result, bitsize, genes, individuals, factor, bias, valmat);

        // print the size of the matrices and the values
    printf("Size of valmat: %d x %d\n", individuals, genes);
    printf("Size of result: %d x %d\n", individuals, genes * bitsize);

    printf("valmat after: \n");
    printfMatrix(valmat, individuals, genes, 6);

    printf("\n");

    // print the difference between the matrices 
    printf("Difference between valmat and copyvalmat: \n");
    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            printf("%f ", valmat[i][j] - copyvalmat[i][j]);
        }
        printf("\n");
    }

    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            if(roundf(valmat[i][j]) == roundf(copyvalmat[i][j])){
                same_count++;
            }
        }
    }

    printf("same_count: %d\n", same_count);

    if (same_count == individuals * genes){
        printf("Test passed\n");
    } else {
        printf("Test failed\n");
    }

    free(valmat);
    free(result);
    free(copyvalmat);
    free(same);

}