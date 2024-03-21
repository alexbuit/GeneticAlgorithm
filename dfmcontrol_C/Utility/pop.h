
void bitpop(int bitsize, int genes, int individuals, int** result);
void bitpop32(int genes, int individuals, int** result);

void uniform_bit_pop(int bitsize, int genes, int individuals, double factor, double bias, int** result);
void normal_bit_pop(int bitsize, int genes, int individuals, double factor, double bias, double loc, double scale, int** result);
void normal_bit_pop_boxmuller(int bitsize, int genes, int individuals, double factor, double bias, double loc, double scale, int** result);
void cauchy_bit_pop(int bitsize, int genes, int individuals, double factor, double bias, double loc, double scale, int** result);

