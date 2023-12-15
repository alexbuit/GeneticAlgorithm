void bitpop(int bitsize, int genes, int individuals, int** result);
void uniform_bit_pop(int bitsize, int genes, int individuals, float factor, float bias, int** result);
void normal_bit_pop(int bitsize, int genes, int individuals, float factor, float bias, float loc, float scale, int** result);
void normal_bit_pop_boxmuller(int bitsize, int genes, int individuals, float factor, float bias, float loc, float scale, int** result);
void cauchy_bit_pop(int bitsize, int genes, int individuals, float factor, float bias, float loc, float scale, int** result);