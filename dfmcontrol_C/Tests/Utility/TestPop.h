void Testbitpop(int bitsize, int genes, int individuals, int writeresult);
void Testuniformpop(int bitsize, int genes, int individuals,
                     double factor, double bias, int writeresult);
void TestnormalpopBM(int bitsize, int genes, int individuals, double loc, double scale,
                     double factor, double bias, int writeresult);
void Testnormalpop(int bitsize, int genes, int individuals, double loc, double scale,
                     double factor, double bias, int writeresult);
void write2file(int bitsize, int genes, int individuals,double factor, double bias, char* filename, int** result, double** numresult);
void Testcauchypop(int bitsize, int genes, int individuals, double loc, double scale,
                     double factor, double bias, int writeresult);
