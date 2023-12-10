
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../../Helper/Helper.c"
#include "TestErrorRate.h"

float** valmat;
int** result;
float** copyvalmat;

float factor = 15;
float bias = 5;
int normalised = 1;

int main(){

    int individuals = 10000;
    int genes = 65;
    
    int bitstepsize = 1;
    int bitstart = 4;
    int bitrange = 48;

    // printf("Error rate: %f\n", ErrorRate(16, genes, individuals));
    // printf("Error rate: %f\n", ErrorRate(32, genes, individuals));
    // printf("Error rate: %f\n", ErrorRate(64, genes, individuals));

    int* bitarray;
    float* error_rate;

    bitarray = malloc((int) (roundf((bitrange * sizeof(int))/ bitstepsize)) - bitstart);
    error_rate = malloc((int) (roundf((bitrange * sizeof(int))/ bitstepsize)) - bitstart);
    


    for(int i=bitstart; i<bitrange; i+=bitstepsize){
        printf("i: %d\n", i);
        bitarray[(int) roundf(i / bitstepsize)] = i;
        error_rate[(int) roundf(i / bitstepsize)] = ErrorRate(i, genes, individuals);
    }

    // write the results to a file
    FILE *fp;
    fp = fopen("error_rate_zoom100x1000_c.txt", "w");
    fprintf(fp, "bitsize error_rate\n");
    for(int i=4; i<bitrange; i++){
        fprintf(fp, "%d %f\n", bitarray[i], error_rate[i]);
    }
    fclose(fp);


    return 0;

}

float ErrorRate(int bitsize, int genes, int individuals){

    valmat = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of integers
    result = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of bitarrays
    copyvalmat = malloc(individuals * sizeof(int*)); // copy of valmat

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
            valmat[i][j] = rand() % 5 * sign; // random integer between -2^bitsize-1 and 2^bitsize - 1
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
    int2ndbit(valmat, bitsize, genes, individuals, factor, bias, normalised, result);

    // Check the result by doing the reverse operation
    int** same = malloc(individuals * sizeof(int*));
    for(int i=0; i<individuals; i++){
        same[i] = malloc(genes * sizeof(int));
    }

    int same_count = 0;

    ndbit2int(result, bitsize, genes, individuals, factor, bias, normalised, valmat);

    for(int i=0; i<individuals; i++){
        for(int j=0; j<genes; j++){
            if(roundf(valmat[i][j]) == roundf(copyvalmat[i][j])){
                same_count++;
            }
        }
    }

    printf("same_count: %d\n", (individuals * genes) - same_count);

    if (same_count == individuals * genes){
        printf("Test passed\n");
    } else {
        printf("Test failed\n");
    }

    free(valmat);
    free(result);
    free(copyvalmat);
    free(same);

    float error_rate = (float) ((individuals * genes) - same_count) / (individuals * genes);
    return error_rate;
}