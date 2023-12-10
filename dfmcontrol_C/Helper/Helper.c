
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

    // 

}

void int2bin(int value, int bitsize, int* result){
    /*
    Convert an integer to a bitarray
    
    valuel is the integer to be converted to a bitarray
    bitsize is the size of the bitarray
    result is the bitarray to be filled with the converted values
    */

    // first bit is the sign bit
    if (value < 0){
        result[0] = 1;
        value = value * -1;
    }
    else {
        result[0] = 0;
    }

    // convert the value to a bitarray
    for (int i = 1; i < bitsize; i++){
        result[i] = value % 2;
        value = value / 2;
    }
}

void intarr2binarr(int* valarr, int bitsize, int genes, int* result){
    /*

    Convert an array of integers to an array of bitarrays

    valarr is the array of integers to be converted to bitarrays (a)
    bitsize is the size of the bitarrays
    genes is the number of genes in the bitarrays (n = genes / bitsize; n = a / bitsize)
    result is the array of bitarrays to be filled with the converted values (n * bitsize)

    */

    // convert the values to bitarrays
    for (int i = 0; i < genes; i++){
        int2bin(valarr[i], bitsize, &result[i * bitsize]);
    }
    
}


void intmat2binmat(int** valmat, int bitsize, int genes, int individuals, int** result){
    /* 
    
    Convert a matrix of integers to a matrix of bitarrays (a x b) (individuals x genes)

    valmat is the matrix of integers to be converted to bitarrays (a x b) (individuals x genes)
    bitsize is the size of the bitarrays
    genes is the number of genes in the bitarrays (n = genes * bitsize; n = b * bitsize)
    individuals is the number of individuals in the bitarrays (m = individuals; m = a)
    result is the matrix of bitarrays to be filled with the converted values (m x n)
    
    */

    // convert the values to bitarrays
    for (int i = 0; i < individuals; i++){
        intarr2binarr(valmat[i], bitsize, genes, &result[i * genes * bitsize]);
    }
}

void bin2int(int* value, int bitsize, int* result){

    // check if the dimensions are correct
    // if (sizeof(result) != bitsize){
    //     printf("Error: result and bitsize have different dimensions\n");
    //     exit(1);
    // }

    // convert the bitarray to an integer

    int sign = 1;
    int res = 0;

    if (value[0] == 1){
        sign = -1;
    }
    else {
        sign = 1;
    }

    for (int i = 1; i < bitsize; i++){
        res += value[i] * pow(2, i - 1);
    }
    res = res * sign;

    *result = res;


}


void binarr2intarr(int* value, int bitsize, int genes, int* result){
    
    // check if the dimensions are correct
    // if (sizeof(result) / sizeof(result[0]) != (sizeof(value) / sizeof(value[0])) / bitsize){
    //     printf("Error: result and size have different dimensions\n");
    //     printf("sizeof(result): %d\n", sizeof(result) / sizeof(result[0]));
    //     printf("sizeof(value): %d\n", sizeof(value) / sizeof(value[0]) / bitsize);
    //     exit(1);
    // }

    // convert the values to integers
    for(int i = 0; i < genes; i++){
        for(int j = 0; j < bitsize; j++){
            bin2int(&value[i * bitsize], bitsize, &result[i]);
        }
    }
}

void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result){

    /*
    Convert a matrix of bitarrays to a matrix of integers
    
    valmat is the matrix of bitarrays to be converted to integers (a x b) (individuals x (bitsize * genes))
    bitsize is the size of the bitarrays
    genes is the number of genes in the bitarrays (n = genes / bitsize; n = b / bitsize)
    individuals is the number of individuals in the bitarrays (m = individuals; m = a)
    result is the matrix of integers to be filled with the converted values (m x n)
    
    */

    // define the conversion matrix
    int* a = malloc(genes * bitsize * sizeof(int*)); 
    int** tempmat = malloc(individuals * sizeof(int*));
    int** sign = malloc(bitsize * sizeof(int));

    for (int i = 0; i < individuals; i++){
        tempmat[i] = malloc(genes * (bitsize - 1) * sizeof(int));
    }

    // fill tempmat with the values of valmat starting at the 1st column of each gene
    for (int i = 0; i < individuals; i++){ // all rows
        for (int j = 0; j < genes; j++){ // all columns
            for (int k = 1; k < bitsize; k++){ // all genes without the sign bit
                tempmat[i][j * (bitsize - 1) + k - 1] = valmat[i][j * bitsize + k];
            }
        }
    }

    // fill the sign array using the first column of valmat of each gene
    for (int i = 0; i < individuals; i++){
        sign[i] = malloc(genes * sizeof(int));
    }

    for (int i = 0; i < individuals; i++){
        for (int j = 0; j < genes; j++){
            sign[i][j] = valmat[i][j * bitsize];
        }
        // if the sign is negative -1 else 1
        for (int j = 0; j < individuals; j++){
            if (sign[i][j] == 1){
                sign[i][j] = -1;
            }
            else {
                sign[i][j] = 1;
            }
        }
    }

    for (int i = 0; i < genes * bitsize; i++){
        a[i] = pow(2, i);
    }

    printf("tempmat: \n");
    printMatrix(tempmat, individuals, genes * (bitsize - 1));
    printf("a: \n");
    printMatrix(a, genes * bitsize, 1);
    printf("sign: \n");
    printMatrix(sign, individuals, genes);

    // convert the values with valmat @ a
    for (int i = 0; i < individuals; i++){
        for (int j = 0; j < genes; j++){
            for (int k = 0; k < bitsize-1; k++){
                result[i][j] += tempmat[i][j + k] * a[k];
            }
            result[i][j] = result[i][j] * sign[i][j];
        }
    }

    free(tempmat);
    free(a);
}

void printMatrix(int** matrix, int rows, int cols) {
    printf("cols: %d\n", cols);
    printf("rows: %d\n", rows);

    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%d", matrix[i][j]);
            if (j < cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < rows - 1) {
            printf(", \n");
        }
    }
    printf("]\n");
}