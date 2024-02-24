

const int flat_linear = 0;
const int flat_exponential = 1;
const int flat_logarithmic = 2;
const int flat_normalized = 3;
const int flat_sigmoid = 4;
const int flat_none = 5;

// Selection functions
const int sel_roulette = 0;
const int sel_rank_tournament = 1;
const int sel_rank = 2;
const int sel_rank_space = 3;
const int sel_boltzmann = 4;


// gen purpose
void process_fx(double** pop, int individuals, int genes, double(*fx)(double*), double* result);
void process_flatten(double* pop, int individuals, int flatten_method, int mode, double flatten_factor, double flatten_bias, double* result);
void process_selection(double* result,int individuals,int selection_method,double selection_div_param,double selection_prob_param, double selection_temp_param, int* selected);


// Flattening functions
void lin_flattening(double* pop, int individuals, double a, double b,double* result);
void exp_flattening(double* pop, int individuals, double a, double b,double* result);
void log_flattening(double* pop, int individuals, double a, double b,double* result);
void norm_flattening(double* pop, int individuals, double a, double b,double* result);
void sig_flattening(double* pop, int individuals, double a, double b,double* result);
void no_flattening(double* pop, int individuals, double a, double b,double* result);


// Selection functions
void roulette(double* pop,int individuals, int genes, int** result);
void rank_tournament_selection(double* pop, int individuals, int genes, int tournament_size, double prob_param, int** result);
void rank_selection(double* pop, int individuals, int genes, double prob_param, int** result);
void rank_space_selection(double* pop, int individuals, int genes, double prob_param, double div_param, int** result);
void boltzmann_selection(double* pop, int individuals, int genes, double temp_param, int** result);

