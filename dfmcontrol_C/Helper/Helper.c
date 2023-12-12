
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "Helper.h"

// Remove the duplicate function declaration
void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                float factor, float bias, int normalised, float** result){
    /*
    Convert an array of bitarrays to an array of floats

    :param valarr: The array of binary data to be converted to floats (a x b) (individuals x (bitsize * genes))
    :type valarr: array of floats (float **)

    :param bitsize: The size of the bitarrays
    :type bitsize: int

    :param genes: The number of genes in the bitarrays (n = length of a row / bitsize; n = a / bitsize)
    :type genes: int

    :param individuals: the number of individuals in the bitarrays (m = individuals; m = a)
    :type individuals: int
    
    :param result: The array of floats to be filled with the converted values (m x n)
    :type result: array of ints (float **)

    :param factor: The factor of the uniform distribution.
    :type factor: float

    :param bias: The bias of the uniform distribution.
    :type bias: float

    :param normalised: If normalised is 1, the values will be normalised to the range [0, 1] multiplied by factor and added bias
                       If normalised is 0, the values will be normalised to the range [0, 2^bitsize - 1]
    :type normalised: int

    :return: void
    :rtype: void
    */

    // temp int array to store the values
    int** temp = (int**)malloc(individuals * sizeof(int*));

    for (int i = 0; i < individuals; i++){
        temp[i] = (int*)malloc(genes * sizeof(int));
    }


    // convert the values to integers
    binmat2intmat(valarr, bitsize, genes, individuals, temp);

    // normalise the values and apply the factor and bias
    if (normalised == 1){
        for (int i = 0; i < individuals; i++){
            for (int j = 0; j < genes; j++){
                result[i][j] = (float) temp[i][j] / (pow(2, bitsize - 1)) * factor + bias;
            }
        }
    }
    else if (normalised == 0)
    {
        // cast to floats
        for (int i = 0; i < individuals; i++){
            for (int j = 0; j < genes; j++){
                result[i][j] = (float) temp[i][j];
            }
        }
    }

    // free the temp array
    for (int i = 0; i < individuals; i++){
        free(temp[i]);
    }
    free(temp);
}

void int2ndbit(float** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int normalised, int** result){

    /*
    Convert an array of integers to an array of bitarrays

    :param valarr: The array of integers to be converted to bitarrays (a)
    :type valarr: array of floats (float **)

    :param bitsize: The size of the bitarrays
    :type bitsize: int

    :param genes: The number of genes in the bitarrays (n = genes * bitsize; n = a * bitsize)
    :type genes: int

    :param individuals: the number of individuals in the bitarrays (m = individuals; m = a)
    :type individuals: int
    
    :param result: The array of bitarrays to be filled with the converted values (m x n)
    :type result: array of ints (int **)

    :param factor: The factor of the uniform distribution.
    :type factor: float

    :param bias: The bias of the uniform distribution.
    :type bias: float

    :param normalised: If normalised is 1, the values will be normalised to the range [0, 1] multiplied by factor and added bias
                       If normalised is 0, the values will be normalised to the range [0, 2^bitsize - 1]
    :type normalised: int

    :return: void
    :rtype: void
    */

   // create a copy of valarr for integer conversion
    int **copyvalarr = (int**)malloc(individuals * sizeof(int*));

    for (int i = 0; i < individuals; i++){
        copyvalarr[i] = (int*)malloc(genes * sizeof(int));
    }

    // normalise the values and apply the factor and bias and cast to integers
    if (normalised == 1){
        for (int i = 0; i < individuals; i++){
            for (int j = 0; j < genes; j++){
                copyvalarr[i][j] = (int) round((valarr[i][j] - bias) / factor * pow(2, bitsize - 1));

            }
        }
    }


    // convert the values to bitarrays
    intmat2binmat(copyvalarr, bitsize, genes, individuals, result);

    // free the copyvalarr array
    for (int i = 0; i < individuals; i++){
        free(copyvalarr[i]);
    }

    free(copyvalarr);
}

void int2bin(int value, int bitsize, int* result){
    /*
    Convert an integer to a bitarray
    
    :param value: The integer to be converted to a bitarray
    :type value: int

    :param bitsize: is the size of the bitarray
    :type bitsize: int

    :param result: is the bitarray to be filled with the converted values
    :type result: array of ints (int *)
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

    :param valarr: The array of integers to be converted to bitarrays (a)
    :type valarr: array of ints (int *)

    :param bitsize: The size of the bitarrays
    :type bitsize: int

    :param genes: The number of genes in the bitarrays (n = genes / bitsize; n = a / bitsize)
    :type genes: int

    :param result: The array of bitarrays to be filled with the converted values (n * bitsize)
    :type result: array of ints (int *)
    
    :return: void
    */

    // convert the values to bitarrays
    for (int i = 0; i < genes; i++){
        int2bin(valarr[i], bitsize, &result[i * bitsize]);
    }
    
}


void intmat2binmat(int** valmat, int bitsize, int genes, int individuals, int** result){
    /* 
    
    Convert a matrix of integers to a matrix of bitarrays (a x b) (individuals x genes)

    :param valmat: The matrix of integers to be converted to bitarrays (a x b) (individuals x genes)
    :type valmat: array of ints (int **)
    
    :param bitsize: The size of the bitarrays
    :type bitsize: int
    
    :param genes: The number of genes in the bitarrays (n = genes * bitsize; n = b * bitsize)
    :type genes: int

    :param individuals: The number of individuals in the bitarrays (m = individuals; m = a)
    :type individuals: int

    :param result: The matrix of bitarrays to be filled with the converted values (m x n)
    :type result: array of ints (int **)

    */

    // convert the values to bitarrays
    for (int i = 0; i < individuals; i++){
        intarr2binarr(valmat[i], bitsize, genes, result[i]);
    }
}

int bin2int(int* value, int bitsize){

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

    return res;
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
        result[i] = bin2int(&value[i * bitsize], bitsize);
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

    for (int i = 0; i < individuals; i++){
        binarr2intarr(valmat[i], bitsize, genes, result[i]);
    }

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

void printfMatrix(float** matrix, int rows, int cols) {
    printf("cols: %d\n", cols);
    printf("rows: %d\n", rows);

    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
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

void sigmoid(float* x, float* result, int size){
    /*
    Calculate the sigmoid of x

    x is the input
    result is the output
    */

    for (int i = 0; i < size; i++){
        result[i] = 1 / (1 + exp(-x[i]));
    }
}

void sigmoid_derivative(float* x, float* result, int size){
    /*
    Calculate the derivative of the sigmoid of x

    x is the input
    result is the output
    */

    for (int i = 0; i < size; i++){
        result[i] = x[i] * (1 - x[i]);
    }
}

void sigmoid2(float* x, float a, float b, float c, float d, float Q, float nu ,float* result, int size){

    /*
    Calculate the sigmoid of x

    x is the input
    result is the output
    */

    for (int i = 0; i < size; i++){
        result[i] = a + (b - a) / (1 + Q * pow(exp(-c * (x[i] - d)), (1/nu)));
    }
}