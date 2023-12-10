void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int normalised, int** result);
void int2ndbit(int** valarr, int bitsize, int genes, int individuals,
               float factor, float bias, int normalised, int** result);
void int2bin(int valuel, int bitsize, int* result);
void bin2int(int* value, int bitsize, int* result);
void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result);