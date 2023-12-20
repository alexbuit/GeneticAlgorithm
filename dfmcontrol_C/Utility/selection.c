#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "selection.h"

// gen purpose
void calc_fx(float** pop, int individuals, int genes, float(*fx)(float), float** result){

}

// Flattening functions
void lin_flattening(float* pop, int individuals, float a, float b,float* result){

    float sum;

    for(int i = 0; i< individuals; i++){
            sum += pop[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = (pop[i]/sum) * a + b;
    }

}
void exp_flattening(float* pop, int individuals, float a, float b,float* result){

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
void norm_flattening(float* pop, int individuals, float a, float b,float* result){

    float sum;

    for(int i = 0; i< individuals; i++){
            sum += pop[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = pop[i]/sum;
    }

}
void sig_flattening(float* pop, int individuals,float a, float b,float* result){

}
void no_flattening(float* pop, int individuals, float a, float b, float* result){

}


// Selection functions
void roulette(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), int mode, float a, float b,int** result){
        /*

        :param pop: matrix of individuals as float (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (float array x0 x1 ... xn)
        :param flatten: flattening function (float array, int, float, float, float array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

        float* fitness = malloc(individuals * sizeof(float));

        for(int i = 0; i< individuals; i++){
            fitness[i] = fx(pop[i][0]);
        }

        flatten(fitness, individuals, a, b, fitness);

        float sum;

        for(int i = 0; i< individuals; i++){
            sum += fitness[i];
        }

        // compute probabilities
        float* prob = malloc(individuals * sizeof(float));

        for(int i = 0; i< individuals; i++){
            if(mode == 1){ // optimisation
                prob[i] = fitness[i]/sum;
        }
        else if(mode == 0){ // minimisation
                prob[i] = 1 - fitness[i]/sum;
        }
        }
        int* selected = malloc(individuals * sizeof(int));
        // select individuals
        roulette_wheel(prob, individuals, individuals, selected);

        // make pairs
        for(int i = 0; i< (int) ceilf(individuals/2); i+=2){
                result[i][0] = selected[i];
                result[i][1] = selected[i+1];
        }
}
void rank_tournament_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), int tournament_size, float prob_param, int mode, float a, float b, int** result){
        /*

        :param pop: matrix of individuals as float (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (float array x0 x1 ... xn)
        :param flatten: flattening function (float array, int, float, float, float array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
void rank_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), float prob_param, int mode, float a, float b, int** result){
        /*

        :param pop: matrix of individuals as float (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (float array x0 x1 ... xn)
        :param flatten: flattening function (float array, int, float, float, float array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
void rank_space_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), float prob_param, float div_param, int mode, float a, float b, int** result){
        /*

        :param pop: matrix of individuals as float (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (float array x0 x1 ... xn)
        :param flatten: flattening function (float array, int, float, float, float array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
void boltzmann_selection(float** pop, int individuals, int genes, float(*fx)(float), float(*flatten)(float*, int, float, float, float*), float temp_param, int mode, float a, float b, int** result){
            /*

        :param pop: matrix of individuals as float (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (float array x0 x1 ... xn)
        :param flatten: flattening function (float array, int, float, float, float array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
