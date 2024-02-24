
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#include "benchmark_conversion.h"

double* runtime_int2bit;
double* runtime_bit2int;


int main(){

    runtime_int2bit = malloc(40 * sizeof(double));
    runtime_bit2int = malloc(40 * sizeof(double));

    printf("start\n");
    for (int i = 1; i < 40; i++){
        matrix_conversion_benchmark(16, i, 16);
    }

    // write the results to a file
    FILE *fp;
    fp = fopen("conversion_benchmarks.txt", "w");
    fprintf(fp, "genes int2bit bit2int\n");
    for(int i=1; i<40; i++){
        fprintf(fp, "%d %f %f\n", i, runtime_int2bit[i], runtime_bit2int[i]);
    }
    fclose(fp);


    // free the memory
    free(runtime_int2bit);
    free(runtime_bit2int);

}

void matrix_conversion_benchmark(int individual, int genes, int bitsize){

    // initialise valmat
    double** valmat = (double**) malloc(individual * sizeof(double*));
    for (int i = 0; i < individual; i++){
        valmat[i] = (double*) malloc(genes  * sizeof(double));
    }

    // initialise result matrix
    int** result = (int**)malloc(individual * sizeof(int*));
    for (int i = 0; i < individual; i++){
        result[i] = (int*)malloc(genes * bitsize * sizeof(int));
    }

    bitpop(bitsize, genes, individual, result);


    // time the conversion from int to bitarray
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    for(int i=0; i<1000; i++){
        ndbit2int32(result, genes, individual, 5.0f, 0.0f, valmat);
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    runtime_bit2int[genes] = cpu_time_used;

    printf("runtime_int2bit[%d]: %f\n", genes, runtime_bit2int[genes]);


    // time the conversion from bitarray to int
    start = clock();

    for(int i=0; i<1000; i++){
        int2ndbit32(valmat, genes, individual, 5.0f, 0.0f, result);
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    runtime_int2bit[genes] = cpu_time_used;

    printf("runtime_int2bit[%d]: %f\n", genes, runtime_int2bit[genes]);
    
    // free the memory
    for (int i = 0; i < individual; i++){
        free(valmat[i]);
        free(result[i]);
    }
    
    free(valmat);
    free(result);
}
