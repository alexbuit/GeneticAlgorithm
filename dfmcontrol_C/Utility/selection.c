#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "selection.h"

// gen purpose
void calc_fx(float** pop, int individuals, int genes, float(*fx)(float), float** result){

}

// Flattening functions
void lin_flattening(float* pop, int individuals, int genes, float a, float b,float* result){

    float sum;

    for(int i = 0; i< individuals; i++){
            sum += pop[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = (pop[i]/sum) * a + b;
    }

}
void exp_flattening(float* pop, int individuals, int genes, float a, float b,float* result){

    float sum;

    for(int i = 0; i< individuals; i++){
            sum += pop[i];
    }

    for(int i = 0; i< individuals; i++){ 
            result[i] = exp((pop[i]/sum) * a) + b;
    }

}
void log_flattening(float* pop, int individuals, float a, float b,float* result){

    /*
    
    Compute:

    .. math::
        f(x) = a^\log(\dfrac{x}{\mathrm{sum}(x)) + b

    For all fitness values in pop (individuals)

    :param pop: matrix of fitness values or 

    */
    
    float sum;

    for(int i = 0; i< individuals; i++){
            sum += pop[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = logf((pop[i]/sum))/logf(a) + b;
    }
}
void norm_flattening(float* pop, int individuals, int genes, float a, float b,float* result){

    float sum;

    for(int i = 0; i< individuals; i++){
            sum += pop[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = pop[i]/sum;
    }

}
void sig_flattening(float* pop, int individuals, int genes, float a, float b,float* result){

}
void no_flattening(float* pop, int individuals, int genes, float a, float b,float* result){

}


// Selection functions
void roulette(float** pop, int individuals, int genes, float(*fx)(float),int mode, int** result){

}
void rank_tournament_selection(float** pop, int individuals, int genes, float(*fx)(float), int tournament_size, float prob_param, int mode, int** result){

}
void rank_selection(float** pop, int individuals, int genes, float(*fx)(float), float prob_param, int mode, int** result){

}
void rank_space_selection(float** pop, int individuals, int genes, float(*fx)(float), float prob_param, float div_param, int mode, int** result){

}
void boltzmann_selection(float** pop, int individuals, int genes, float(*fx)(float), float temp_param, int mode, int** result){
    
}
