#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.h"
#include "flatten.h"



void process_flatten(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){
        /*

        :param pop: matrix of fitness values or 

        */
        // // copy the 

        // if(gene_pool.fx_param.fx_optim_mode == 0){
        //         for(int i = 0; i< gene_pool.individuals; i++){
        //                 pop_result_set[i] = -pop_result_set[i];
        //         }

        // }
        // else{
        //         printf("Error: mode is not 0 or 1\n");
        //         exit(1);
        // }
        

        if(flatten_param.flatten_method==flat_linear){
        lin_flattening(gene_pool, flatten_param, result);
        }
        else if(flatten_param.flatten_method==flat_exponential){
        exp_flattening(gene_pool, flatten_param, result);
        }
        else if(flatten_param.flatten_method==flat_logarithmic){
        log_flattening(gene_pool, flatten_param, result);
        }
        else if(flatten_param.flatten_method==flat_normalized){
        norm_flattening(gene_pool, flatten_param, result);
        }
        else if(flatten_param.flatten_method==flat_sigmoid){
        sig_flattening(gene_pool, flatten_param, result);
        }
        else if(flatten_param.flatten_method==flat_none){
        no_flattening(gene_pool, flatten_param, result);
        }
        else{
        printf("Error: flatten_method is not 0, 1, 2, 3, 4 or 5\n");
        }

}

// Flattening functions

// Struct meegegeven
void lin_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){

    double sum;

    for(int i = 0; i< gene_pool.individuals; i++){
                // hier min max functie
            sum += gene_pool.pop_result_set[i];
    }

    for(int i = 0; i< gene_pool.individuals; i++){
            result[i] = (gene_pool.pop_result_set[i]/sum) *flatten_param.flatten_factor + flatten_param.flatten_bias;
    }

}
void exp_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){

    double sum;

    for(int i = 0; i< gene_pool.individuals; i++){
            sum += gene_pool.pop_result_set[i];
    }

    for(int i = 0; i< gene_pool.individuals; i++){ 
            result[i] = exp((gene_pool.pop_result_set[i]/sum) *flatten_param.flatten_factor) + flatten_param.flatten_bias;
    }

}
void log_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){

    /*
    
    Compute:

    .. math::
        f(x) = a^\log(\dfrac{x}{\mathrm{sum}(x)) + b

    For all fitness values in pop (individuals)

    :param pop: matrix of fitness values or 

    */
    
    double sum;

    for(int i = 0; i< gene_pool.individuals; i++){
            sum += gene_pool.pop_result_set[i];
    }
    double lgda = log(flatten_param.flatten_factor);
    for(int i = 0; i< gene_pool.individuals; i++){
            result[i] = log((gene_pool.pop_result_set[i]/sum))/lgda + flatten_param.flatten_bias;
    }
}
void norm_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){

    double sum;

    for(int i = 0; i< gene_pool.individuals; i++){
            sum += gene_pool.pop_result_set[i];
    }

    for(int i = 0; i< gene_pool.individuals; i++){
            result[i] = gene_pool.pop_result_set[i]/sum;
    }

}
void sig_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){

}
void no_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result){

}

