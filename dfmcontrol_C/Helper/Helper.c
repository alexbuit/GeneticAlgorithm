
#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "Helper.h"

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

    /*

    Convert a bitarray to an integer

    :param value: The bitarray to be converted to an integer
    :type value: array of ints (int *)

    :param bitsize: The size of the bitarray
    :type bitsize: int

    :return: The integer value of the bitarray
    :rtype: int

    */

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
    
    /*
    Convert an array of bitarrays to an array of integers

    :param valarr: The array of bitarrays to be converted to integers (a)
    :type valarr: array of ints (int *)

    :param bitsize: The size of the bitarrays
    :type bitsize: int

    :param genes: The number of genes in the bitarrays (n = genes / bitsize; n = a / bitsize)
    :type genes: int

    :param result: The array of integers to be filled with the converted values (n)
    :type result: array of ints (int *)

    :return: void
    */


    // convert the values to integers
    for(int i = 0; i < genes; i++){
        result[i] = bin2int(&value[i * bitsize], bitsize);
    }
}

void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result){

    /*
    Convert a matrix of bitarrays to a matrix of integers (a x b) (individuals x genes)

    :param valmat: The matrix of bitarrays to be converted to integers (a x b) (individuals x genes)
    :type valmat: array of ints (int **)

    :param bitsize: The size of the bitarrays
    :type bitsize: int

    :param genes: The number of genes in the bitarrays (n = genes * bitsize; n = b * bitsize)
    :type genes: int

    :param individuals: The number of individuals in the bitarrays (m = individuals; m = a)
    :type individuals: int

    :param result: The matrix of integers to be filled with the converted values (m x n)
    :type result: array of ints (int **)
    */

    for (int i = 0; i < individuals; i++){
        binarr2intarr(valmat[i], bitsize, genes, result[i]);
    }

}

void printMatrix(int** matrix, int rows, int cols) {

    /*

    Print a matrix of integers

    :param matrix: The matrix to be printed
    :type matrix: array of ints (int **)

    :param rows: The number of rows in the matrix
    :type rows: int

    :param cols: The number of columns in the matrix
    :type cols: int

    :return: void

    */

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

    /*
    Print a matrix of floats

    :param matrix: The matrix to be printed
    :type matrix: array of floats (float **)

    :param rows: The number of rows in the matrix
    :type rows: int

    :param cols: The number of columns in the matrix
    :type cols: int

    :return: void

    */
   

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

void uniform_random(int m, int n,int lower, int upper,int** result){
    /*
    Create a matrix filled with uniformly distributed integers.

    :param m: Amount of rows
    :type m: int
    :param n: Amount of cols
    :type n: int

    :param lower: Lower bound of the distribution
    :type lower: int
    :param upper: Upper bound of the distribution
    :type upper: int

    :param result: Result matrix to which the distibutution is written to
    :type result: **int (m x n)

    :return: void
    */

    // the range in which the numbers can be generated
    unsigned int nRange = (unsigned int)(upper - lower);
    // The amount of bits generated for a number
    unsigned int nRangeBits = (unsigned int) ceil(log2((double) (nRange)));

    unsigned int nRand; // random number


    // for the matrix write uniformly distributed numbers
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            do{
                nRand = 0;
                for(int k=0; k<nRangeBits; k++){
                    nRand = (nRand << 1) | (rand() & 1); // lshift and rand or 1
                }
            } while(nRand >= nRange);
            result[i][j] = (int) (nRand + lower);
        }
    }


}