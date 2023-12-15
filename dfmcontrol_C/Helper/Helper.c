
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdint.h"

#include "Helper.h"
#undef PI
#define PI   3.14159265358979323846264338327950288419716939937510f

void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                float factor, float bias, float** result){
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

    :return: void
    :rtype: void
    */

    int temp;

    for (int i = 0; i < individuals; i++){
        for (int j = 0; j < genes; j++){
            // result[i][j] = (float) temp[i][j] / (pow(2, bitsize - 1)) * factor + bias;
            if(valarr[i][j] < 0){
                temp = ~(valarr[i][j] & 0x7fffffff) + 1 ;
            }
            else{
                temp = valarr[i][j];
            }
            result[i][j] = (float) temp * factor / (pow(2, bitsize - 1)) + bias;
        }
    }
}

void int2ndbit(float** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int** result){

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

    :return: void
    :rtype: void
    */

    // normalise the values and apply the factor and bias and cast to integers
    int temp;
    for (int i = 0; i < individuals; i++){
        for (int j = 0; j < genes; j++){
            temp = (int) roundf((valarr[i][j] - bias) * pow(2, bitsize - 1) / factor);
            if (temp < 0){
                result[i][j] = ~(temp-1) | 0x80000000; // bitflip and subtract 1
            }
            else {
                result[i][j] = temp;
            }
        }
    }

}

void int2bin(int value, int bitsize, int result){
    /*
    Convert an integer to a bitarray
    
    :param value: The integer to be converted to a bitarray
    :type value: int

    :param bitsize: is the size of the bitarray
    :type bitsize: int

    :param result: is the bitarray to be filled with the converted values
    :type result: array of bytes
    */

    // first bit is the sign bit
    if (value < 0){
        result = ~(value-1); // bitflip and subtract 1
    }
    else {
        result = value;
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

void printfMatrix(float** matrix, int rows, int cols, int precision) {

    /*
    Print a matrix of floats

    :param matrix: The matrix to be printed
    :type matrix: array of floats (float **)

    :param rows: The number of rows in the matrix
    :type rows: int

    :param cols: The number of columns in the matrix
    :type cols: int

    :param precision: The number of decimals to be printed
    :type precision: int

    :return: void

    */
   

    printf("cols: %d\n", cols);
    printf("rows: %d\n", rows);

    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%.*f", precision, matrix[i][j]);
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

float gaussian(float x, float mu, float sigma){
    /*
    Calculate the gaussian of x

    x is the input
    mu is the mean
    sigma is the standard deviation
    */

    float result = (1 / (sigma * sqrtf(2 * PI))) * expf(-powf(x - mu, 2) / (2 * powf(sigma, 2)));

    return result;
}

float cauchy(float x, float mu, float sigma){
    /*
    Calculate the cauchy of x

    x is the input
    mu is the mean
    sigma is the standard deviation
    */

    float result = (1 / PI) * (sigma / (powf(x - mu, 2) + powf(sigma, 2)));
    
    return result;
}

void roulette_wheel(double* probabilities, int size, int ressize ,int* result){

    /*
    Roulette wheel selection of an index based on probabilities

    :param probabilities: The probabilities of the indices
    :type probabilities: array of floats (float *)

    :param size: The size of the probabilities array
    :type size: int

    :param ressize: The size of the result array (amount of indices to be selected)
    :type ressize: int

    :param result: The index selected
    :type result: array of ints (int *)

    */

    // create a copy of the probabilities array
    double* copy = (double*)malloc(size * sizeof(double));
    int* indices = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++){
        copy[i] = probabilities[i];
        indices[i] = i;
    }

    // sort the copy array in ascending order and keep track of the indices
    for (int i = 0; i < size; i++){ // expensive sorting algorithm
        for (int j = i + 1; j > size; j++){
            if (copy[i] < copy[j]){
                float temp = copy[i];
                copy[i] = copy[j];
                copy[j] = temp;

                int temp2 = indices[i];
                indices[i] = indices[j];
                indices[j] = temp2;
            }
        }
    }


    // calculate the cumulative sum of the probabilities
    double* cumsum = (double*)malloc(size * sizeof(double));
    cumsum[0] = copy[0];

    for (int i = 1; i < size; i++){
        cumsum[i] = cumsum[i - 1] + copy[i];
    }

    // generate random numbers and select the indices

    for (int i = 0; i < ressize; i++){
        double randnum = (double)rand() / RAND_MAX;

        for (int j = 0; j < size; j++){
            if (randnum < cumsum[j]){
                result[i] = indices[j];
                break;
            }
        }
    }

    // free the arrays
    free(copy);
    free(indices);
    free(cumsum);
}

int random_int(){
    return (rand() % 0x00008000) * 0x00020000 +  (rand() % 0x00008000) * 4 + (rand() % 4);
}