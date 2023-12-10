void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                float factor, float bias, int normalised, float** result);
void int2ndbit(int** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int normalised, int** result);
void int2bin(int value, int bitsize, int* result);
void intarr2binarr(int* valarr, int bitsize, int size, int* result);
void intmat2binmat(int** valmat, int bitsize, int genes, int individuals, int** result);
void bin2int(int* value, int bitsize, int* result);
void binarr2intarr(int* value, int bitsize, int genes, int* result);
void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result);
void printMatrix(int** matrix, int rows, int cols);