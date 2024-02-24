#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.h"
#include "selection.h"


void process_pop(double** pop,  double(*fx)(double*),  int individuals, int genes, int flatten_method, double flatten_factor, double flatten_bias, int flatten_optim_mode,
                 int selection_method, double selection_div_param, double selection_prob_param, double selection_temp_param, int* selected, double* fx_result){
// TODO: check individual even nr 
// TODO: refractor individuals and genes to _count

double* result = malloc(individuals * sizeof(double));


process_fx(pop, individuals, genes, fx, fx_result); // pop, individuals, genes -> struct ?

process_flatten(fx_result, individuals, flatten_method, flatten_optim_mode, flatten_factor, flatten_bias, result);

process_selection(result, individuals, selection_method, selection_div_param, selection_prob_param, selection_temp_param, selected);

free (result);
}

// gen purpose
// void process_pop(){
        // TODO: implement
        // if fx equal booths function then excecute booth function
        // else if fx equal rosenbrock function then excecute rosenbrock function
        // .....

        // "config:"
        // "fx=booths function"
        //  Accepts array of float -> gives float

        // "flatten=linear flattening"
        //  reduces solution space to [-1, 1], with distribution of fitness values 
        //  array of float -> array of float

        // "selection=roulette"
        //  Sorts next gen parents by fitness and selects parents based on their fitness
        //  array of float -> array of float

        // "crossover=single point crossover"
        //  Takes two parents and creates two children by certain method

        // "configpop:"
        // "individuals=100"
        // "genes=10"
        // "bitsize=32"

        // "configselection:"
        // "mode=optimisation"
        // "div_param=exp;fitness;negative"

        // "....."

        // "configoutput:"
        // "output=csv"
        // "pop, selection, mutation....."

        // if flatten equal linear flattening then excecute linear flattening
        // else if flatten equal exponential flattening then excecute exponential flattening
        // .....

        // if selection equal roulette then excecute roulette
        // else if selection equal rank tournament selection then excecute rank tournament selection
        // .....

        // if crossover equal single point crossover then excecute single point crossover
        // else if crossover equal two point crossover then excecute two point crossover
        // .....

        // if mutation equal bit flip mutation then excecute bit flip mutation
        // else if mutation equal swap mutation then excecute swap mutation
        // .....

        // output data to file
// }

void  process_fx(double** pop, int individuals, int genes, double(*fx)(double*), double* result){
        /*

        :param pop: matrix of individuals as double (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (double array x0 x1 ... xn)
        :param result: matrix of fitness values (individuals x 1)

        */

        for(int i = 0; i< individuals; i++){
                result[i] = fx(pop[i]);
        }
}

void process_flatten(double* pop, int individuals, int flatten_method, int mode, double flatten_factor, double flatten_bias, double* result){
        /*

        :param pop: matrix of fitness values or 

        */

        if(mode == 0){
                for(int i = 0; i< individuals; i++){
                        pop[i] = -pop[i];
                }

        }
        else{
                printf("Error: mode is not 0 or 1\n");
                exit(1);
        }
        

        if(flatten_method==flat_linear){
        lin_flattening(pop, individuals, flatten_factor, flatten_bias, result);
        }
        else if(flatten_method==flat_exponential){
        exp_flattening(pop, individuals, flatten_factor, flatten_bias, result);
        }
        else if(flatten_method==flat_logarithmic){
        log_flattening(pop, individuals, flatten_factor, flatten_bias, result);
        }
        else if(flatten_method==flat_normalized){
        norm_flattening(pop, individuals, flatten_factor, flatten_bias, result);
        }
        else if(flatten_method==flat_sigmoid){
        sig_flattening(pop, individuals, flatten_factor, flatten_bias, result);
        }
        else if(flatten_method==flat_none){
        no_flattening(pop, individuals, flatten_factor, flatten_bias, result);
        }
        else{
        printf("Error: flatten_method is not 0, 1, 2, 3, 4 or 5\n");
        }

}

// Flattening functions
void lin_flattening(double* fx_result, int individuals, double a, double b, double* result){

    double sum;

    for(int i = 0; i< individuals; i++){
            sum += fx_result[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = (fx_result[i]/sum) * a + b;
    }

}
void exp_flattening(double* fx_result, int individuals, double a, double b, double* result){

    double sum;

    for(int i = 0; i< individuals; i++){
            sum += fx_result[i];
    }

    for(int i = 0; i< individuals; i++){ 
            result[i] = exp((fx_result[i]/sum) * a) + b;
    }

}
void log_flattening(double* fx_result, int individuals, double a, double b,double* result){

    /*
    
    Compute:

    .. math::
        f(x) = a^\log(\dfrac{x}{\mathrm{sum}(x)) + b

    For all fitness values in pop (individuals)

    :param pop: matrix of fitness values or 

    */
    
    double sum;

    for(int i = 0; i< individuals; i++){
            sum += fx_result[i];
    }
    double lgda = logd(a);
    for(int i = 0; i< individuals; i++){
            result[i] = logd((fx_result[i]/sum))/lgda + b;
    }
}
void norm_flattening(double* fx_result, int individuals, double a, double b,double* result){

    double sum;

    for(int i = 0; i< individuals; i++){
            sum += fx_result[i];
    }

    for(int i = 0; i< individuals; i++){
            result[i] = fx_result[i]/sum;
    }

}
void sig_flattening(double* fx_result, int individuals,double a, double b,double* result){

}
void no_flattening(double* fx_result, int individuals, double a, double b, double* result){

}


void process_selection(double* result,int individuals,int selection_method,double selection_div_param,double selection_prob_param, double selection_temp_param, int* selected){

        if(selection_method==sel_roulette){
        roulette(result, individuals, selected);
        }
        else if(selection_method==sel_rank_tournament){
        rank_tournament_selection(result, individuals, selection_div_param, selection_prob_param, selected);
        }
        else if(selection_method==sel_rank){
        rank_selection(result, individuals, selection_prob_param, selected);
        }
        else if(selection_method==sel_rank_space){
        rank_space_selection(result, individuals, selection_prob_param, selection_div_param, selected);
        }
        else if(selection_method==sel_boltzmann){
        boltzmann_selection(result, individuals, selection_temp_param, selected);
        }
        else{
        printf("Error: selection_method is not 0, 1, 2, 3 or 4\n");
        }

}

// Selection functions
void roulette(double* pop ,int individuals, int genes, int** result){
        /*

        :param pop: matrix of normalised fitness values for the population (individuals x 1)
        :param individuals: number of individuals
        :param genes: number of genes
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */


        int* selected = malloc(individuals * sizeof(int));
        // select individuals
        roulette_wheel(pop, individuals, individuals, selected);

        // make pairs
        for(int i = 0; i< (int) (individuals/2); i+=2){
                result[i] = selected[i];
                result[i+1] = selected[i+1];
        }
}
void rank_tournament_selection(double* pop, int individuals, int genes, int tournament_size, double prob_param, int** result){
        /*

        :param pop: matrix of individuals as double (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (double array x0 x1 ... xn)
        :param flatten: flattening function (double array, int, double, double, double array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
void rank_selection(double* pop, int individuals, int genes, double prob_param, int** result){
        /*

        :param pop: matrix of individuals as double (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (double array x0 x1 ... xn)
        :param flatten: flattening function (double array, int, double, double, double array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
void rank_space_selection(double* pop, int individuals, int genes, double prob_param, double div_param, int** result){
        /*

        :param pop: matrix of individuals as double (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (double array x0 x1 ... xn)
        :param flatten: flattening function (double array, int, double, double, double array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
void boltzmann_selection(double* pop, int individuals, int genes, double temp_param, int** result){
            /*

        :param pop: matrix of individuals as double (individuals x genes)
        :param individuals: number of individuals
        :param genes: number of genes
        :param fx: fitness function (double array x0 x1 ... xn)
        :param flatten: flattening function (double array, int, double, double, double array)
        :param a: parameter for flattening function
        :param b: parameter for flattening function
        :param mode: 0 for minimisation, 1 for maximisation
        :param result: matrix of the indices of selected individuals (individuals x 2)

        */

}
