
#include <stdio.h> 
#include <time.h> 

#include "../../../Utility/crossover.h"
#include "../../../Utility/pop.h"


void crossoverbench(int bitsize, int genes, int individuals, int i);

double* runtime_single;
double* runtime_double;
double* runtime_equal;

int main(){


    runtime_single = malloc(40 * sizeof(double));
    runtime_double = malloc(40 * sizeof(double));
    runtime_equal = malloc(40 * sizeof(double));

    printf("start\n");
    for (int i = 1; i < 39; i++){
        crossoverbench(16, i, 16, i);
    }

    // write the results to a file
    FILE *fp;
    fp = fopen("crossover_benchmarks.txt", "w");
    fprintf(fp, "genes single double equal\n");
    for(int i=1; i<40; i++){
        fprintf(fp, "%d %f %f %f\n", i, runtime_single[i], runtime_double[i], runtime_equal[i]);
    }
    fclose(fp);




    free(runtime_single);
    free(runtime_double);
    free(runtime_equal);

}

void crossoverbench(int bitsize, int genes, int individuals, int i){

    int** valmat1;
    int** valmat2;
    int* result1;
    int* result2;


    valmat1 = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of integers
    valmat2 = malloc(individuals * sizeof(int*)); // matrix of individuals amount of genes amount of integers
    result1 = malloc(bitsize * genes * sizeof(int)); // matrix of individuals amount of genes amount of integers
    result2 = malloc(bitsize * genes * sizeof(int)); // matrix of individuals amount of genes amount of integers
    
    for (int i = 0; i < individuals; i++){
        valmat1[i] = malloc(genes * bitsize * sizeof(int));
        valmat2[i] = malloc(genes * bitsize * sizeof(int));
    }

    // fill the matrices with random values
    bitpop(bitsize, genes, individuals, valmat1);
    bitpop(bitsize, genes, individuals, valmat2);
    

    //print the matrices
    

    // crossover
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    
    for (int j = 0; j < 100000; j++)
    for (int i = 0; i < (int) roundf(individuals/2); i+=2){
        single_point_crossover32(valmat1[i], valmat2[i+1], result1, result2, bitsize * genes);
    }
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    runtime_single[i] = cpu_time_used;

    start = clock();
    for (int j = 0; j < 100000; j++)
    for (int i = 0; i < (int) roundf(individuals/2); i+=2){
        two_point_crossover32(valmat1[i], valmat2[i+1], result1, result2, bitsize * genes);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    runtime_double[i] = cpu_time_used;

    start = clock();
    // iterate for 100 times to get a better average

    for (int j = 0; j < 100000; j++)
    for (int i = 0; i < (int) roundf(individuals/2); i+=2){
        uniform_crossover32(valmat1[i], valmat2[i+1], result1, result2, bitsize * genes);
    }


    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    runtime_equal[i] = cpu_time_used;

    printf("single: %f\n", runtime_single[i]);
    printf("double: %f\n", runtime_double[i]);
    printf("equal: %f\n", runtime_equal[i]);

    for(int i=0; i< genes * bitsize; i++){
        printf("%d ", result1[i]);
    }
    printf("\n");
    for(int i=0; i< genes * bitsize; i++){
        printf("%d ", result2[i]);
    }
    printf("\n");

    printf("iteration: %d\n", i);
    // print the result
    // printf("result:\n");
    // printmat(result, individuals, genes);

    // free the matrices
    for (int i = individuals - 1; i >= 0; i--) {
        free(valmat1[i]);
        free(valmat2[i]);
    }

    free(valmat1);
    free(valmat2);
    free(result1);
    free(result2);
}