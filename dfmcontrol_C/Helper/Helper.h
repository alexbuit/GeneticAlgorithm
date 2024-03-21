

int intXOR32_seed;

struct gene_pool_s {
    int** pop_param_bin;
    int** pop_param_bin_cross_buffer;
    double** pop_param_double;
    double* pop_result_set;
    int* selected_indexes;
    int* sorted_indexes;
    int genes;
    int individuals;
    int elitism;
    int iteration_number;
};

struct selection_param_s {
    int selection_method;
    double selection_div_param;
    double selection_prob_param;
    double selection_temp_param;
    int selection_tournament_size;
};

struct flatten_param_s {
    int flatten_method;
    double flatten_factor;
    double flatten_bias;
    int flatten_optim_mode;
};

struct crossover_param_s {
    int crossover_method;
    double crossover_prob;
};

struct mutation_param_s {
    int mutation_method;
    double mutation_prob;
    double mutation_rate;
};

struct fx_param_s {
    int fx_method;
    int fx_optim_mode;
};

struct config_ga_s{
    struct selection_param_s selection_param;
    struct flatten_param_s flatten_param;
    struct crossover_param_s crossover_param;
    struct mutation_param_s mutation_param;
    struct fx_param_s fx_param;
};

// Convesrion functions
void ndbit2int32(int** valarr, int genes, int individuals,
                double factor, double bias, double** result);
void int2ndbit32(double** valarr, int genes, int individuals,
               double factor, double bias, int** result);
void ndbit2int(int** valarr, int bitsize, int genes, int individuals,
                double factor, double bias, double** result);
void int2ndbit(double** valarr, int bitsize, int genes, int individuals,
               double factor, double bias, int** result);  

// conversion helper functions
void int2bin(int value, int bitsize, int* result);
void intarr2binarr(int* valarr, int bitsize, int size, int* result);
void intmat2binmat(int** valmat, int bitsize, int genes, int individuals, int** result);
int bin2int(int* value, int bitsize);
void binarr2intarr(int* value, int bitsize, int genes, int* result);
void binmat2intmat(int** valmat, int bitsize, int genes, int individuals, int** result);

// Printing functions
void printMatrix(int** matrix, int rows, int cols);
void printfMatrix(double** matrix, int rows, int cols, int precision);

// Mathemathical functions
void sigmoid(double* x, double* result, int size);
void sigmoid_derivative(double* x, double* result, int size);
void sigmoid2(double* x, double a, double b, double c, double d, double Q, double nu ,double* result, int size);
void uniform_random(int m, int n,int lower, int upper, int** result);
double gaussian(double x, double mu, double sigma);


// Roulette wheel selection
void roulette_wheel(double* probabilities, int size, int ressize, int* result);

// sorting functions
void indexed_inv_bubble_sort(double* arr, int* indices, int size);
void indexed_merge_sort(double* arr, int* indices, int size);
void indexed_inv_merge_sort(double* arr, int* indices, int size);

// random 32 bit integer in binary
int random_int32();
void seed_intXOR32();
int random_intXOR32();
int intXORshift32(int a);

// usefull for debugging and eventual conversion to numpy arrays
void convert_int32_to_binary(int** valarr, int genes, int individuals,
                             double factor, double bias);
void convert_binary_to_int32(int** valarr, int genes, int individuals,
                             double factor, double bias);
