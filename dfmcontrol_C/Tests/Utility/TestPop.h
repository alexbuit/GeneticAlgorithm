void Testbitpop(int bitsize, int genes, int individuals, int writeresult);
void Testuniformpop(int bitsize, int genes, int individuals,
                     float factor, float bias, int normalised, int writeresult);
void Testnormalpop(int bitsize, int genes, int individuals, float loc, float scale,
                     float factor, float bias, int normalised, int writeresult);

void write2file(int bitsize, int genes, int individuals, char* filename, int** result, float** numresult);