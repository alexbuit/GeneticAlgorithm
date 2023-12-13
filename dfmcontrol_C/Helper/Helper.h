#ifndef _Helper_
#define _Helper_

void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                float factor, float bias, int normalised, float** result);
void int2ndbit(float** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int normalised, int** result);
void int2bin(int value, int bitsize, int* result);
void intarr2binarr(int* valarr, int bitsize, int size, int* result);
void intmat2binmat(int** valmat, int bitsize, int genes, int individuals, int** result);
int bin2int(int* value, int bitsize);
void binarr2intarr(int* value, int bitsize, int genes, int* result);
void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result);
void printMatrix(int** matrix, int rows, int cols);
void printfMatrix(float** matrix, int rows, int cols, int precision);
void sigmoid(float* x, float* result, int size);
void sigmoid_derivative(float* x, float* result, int size);
void sigmoid2(float* x, float a, float b, float c, float d, float Q, float nu ,float* result, int size);
void uniform_random(int m, int n,int lower, int upper, int** result);
float gaussian(float x, float mu, float sigma);
void roulette_wheel(double* probabilities, int size, int ressize, int* result);
#endif
