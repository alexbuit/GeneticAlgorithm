
#ifndef _Helper_
#define _Helper_

// Convesrion functions
void ndbit2int32(int** valarr, int genes, int individuals,
                float factor, float bias, float** result);
void int2ndbit32(float** valarr, int genes, int individuals,
               float factor, float bias, int** result);
void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                float factor, float bias, float** result);
void int2ndbit(float** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int** result);  

// conversion helper functions
void int2bin(int value, int bitsize, int result);
void intarr2binarr(int* valarr, int bitsize, int size, int* result);
void intmat2binmat(int** valmat, int bitsize, int genes, int individuals, int** result);
int bin2int(int* value, int bitsize);
void binarr2intarr(int* value, int bitsize, int genes, int* result);
void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result);

// Printing functions
void printMatrix(int** matrix, int rows, int cols);
void printfMatrix(float** matrix, int rows, int cols, int precision);

// Mathemathical functions
void sigmoid(float* x, float* result, int size);
void sigmoid_derivative(float* x, float* result, int size);
void sigmoid2(float* x, float a, float b, float c, float d, float Q, float nu ,float* result, int size);
void uniform_random(int m, int n,int lower, int upper, int** result);
float gaussian(float x, float mu, float sigma);

// Roulette wheel selection
void roulette_wheel(double* probabilities, int size, int ressize, int* result);

// random 32 bit integer in binary
int random_int32();
// usefull for debugging and eventual conversion to numpy arrays
void convert_int32_to_binary(int** valarr, int genes, int individuals,
                             float factor, float bias);
void convert_binary_to_int32(int** valarr, int genes, int individuals,
                             float factor, float bias);
#endif
