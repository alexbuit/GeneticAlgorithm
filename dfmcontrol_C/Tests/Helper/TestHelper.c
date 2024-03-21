#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../../Helper/Helper.h"

#include "TestHelper.h"

int main(){
    // Test intarr2binarr

    // int genes = 12;
    // int bitsize = 16;

    // int* valarr = malloc(genes * sizeof(int)); // array of genes amount of integers
    // int* result = malloc(genes * bitsize * sizeof(int)); // array of genes amount of bitarrays
    // int* copyvalarr = malloc(genes * sizeof(int)); // copy of valarr

    // int sign;
    // // fill valarr with random integers
    // for(int i=0; i<genes; i++){
    //     sign = (rand() % 2 == 0) ? 1 : -1;
    //     valarr[i] = rand() % (int) pow(2, bitsize - 1) * sign; // random integer between -2^bitsize-1 and 2^bitsize - 1
    //     copyvalarr[i] = valarr[i];
    // }

    // intarr2binarr(valarr, bitsize, genes, result);

    // printf("valarr (in-between): ");
    // for(int i=0; i<genes; i++){
    //     printf("%d ", valarr[i]);
    // }
    // printf("\n");
    // printf("result (in-between): ");
    // for(int i=0; i<genes * bitsize; i++){
    //     printf("%d ", result[i]);
    // }
    // printf("\n");


    // // Check the result by doing the reverse operation
    // binarr2intarr(result, bitsize, genes, valarr);

    // int same_count = 0;
    // for(int i=0; i<genes; i++){
    //     if(valarr[i] == copyvalarr[i]){
    //         same_count++;
    //     }
    // }

    // printf("valarr: ");
    // for(int i=0; i<genes; i++){
    //     printf("%d ", valarr[i]);
    // }

    // printf("\ncopyvalarr: ");
    // for(int i=0; i<genes; i++){
    //     printf("%d ", copyvalarr[i]);
    // }

    // printf("\nresult: ");
    // for(int i=0; i<genes * bitsize; i++){
    //     printf("%d ", result[i]);
    // }

    // printf("\n");

    // printf("same_count: %d\n", same_count);


    // free(valarr);
    // free(result);
    // free(copyvalarr);
\

    // Test binmat2intmat
    
    int individuals = 4;
    int genes = 4;
    int bitsize = 32;

    int** valmat = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of integers
    int** result = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of bitarrays
    int** copyvalmat = malloc(individuals * sizeof(int*)); // copy of valmat

    for(int i=0; i<individuals; i++){
        valmat[i] = malloc(genes * sizeof(int));
        result[i] = malloc(genes * bitsize * sizeof(int));
        copyvalmat[i] = malloc(genes * sizeof(int));
    }

    // fill valmat with random integers
    int sign;
    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            sign = (rand() % 2 == 0) ? 1 : -1;
            valmat[i][j] = rand() % (int) pow(2, bitsize - 1) * sign; // random integer between -2^bitsize-1 and 2^bitsize - 1
            copyvalmat[i][j] = valmat[i][j];
        }
    }

    // print the size of the matrices
    printf("valmat: %d x %d\n", individuals, genes);
    printf("result: %d x %d\n", individuals, genes * bitsize);




    intmat2binmat(valmat, bitsize, genes, individuals, result);

    // print the matrices in-between
    printf("valmat (in-between): \n");
    printMatrix(valmat, individuals, genes);

    printf("\nresult (in-between): \n");
    for (int i = 0; i < individuals; i++) {
        for (int j = 0; j < genes; j++) {
            printf("%d ", result[i][j]);
        }
    }
    printf("\n");
    
    printMatrix(result, individuals, genes * bitsize);

    printf("\n");

    binmat2intmat(result, bitsize, genes, individuals, valmat);

    // Check the result by doing the reverse operation
    int** same = malloc(individuals * sizeof(int*));
    for(int i=0; i<individuals; i++){
        same[i] = malloc(genes * sizeof(int));
    }

    int same_count = 0;

    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            if(valmat[i][j] == copyvalmat[i][j]){
                same[i][j] = 1;
            } else {
                same[i][j] = 0;
            }
        }
    }

    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            if(same[i][j] == 1){
                same_count++;
            }
        }
    }

    printf("valmat: \n");
    printMatrix(valmat, individuals, genes);

    printf("\ncopyvalmat: \n");

    printMatrix(copyvalmat, individuals, genes);

    printf("\n");

    printf("same_count: %d\n", same_count);

    free(valmat);
    free(result);
    free(copyvalmat);
    // free(same);

}

