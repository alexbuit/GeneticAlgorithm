
#ifndef _Selection_
#define _Selection_

// gen purpose
void calc_fx(float** pop, int individuals, int genes, float(*fx)(float), float** result);

// Flattening functions
void lin_flattening(float* pop, int individuals, float a, float b,float* result);
void exp_flattening(float* pop, int individuals, float a, float b,float* result);
void log_flattening(float* pop, int individuals, float a, float b,float* result);
void norm_flattening(float* pop, int individuals, float a, float b,float* result);
void sig_flattening(float* pop, int individuals, float a, float b,float* result);
void no_flattening(float* pop, int individuals, float a, float b,float* result);


// Selection functions
void roulette(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), int mode, float a, float b, int** result);
void rank_tournament_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), int tournament_size, float prob_param, int mode, float a, float b, int** result);
void rank_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), float prob_param, int mode, float a, float b, int** result);
void rank_space_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), float prob_param, float div_param, int mode, float a, float b, int** result);
void boltzmann_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), float temp_param, int mode, float a, float b, int** result);

#endif