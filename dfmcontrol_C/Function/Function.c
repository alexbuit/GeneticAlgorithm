
#include <math.h>
#include "Function.h"
#include "../Helper/Helper.h"

void process_fx(struct gene_pool_s gene_pool, struct fx_param_s fx_param){
    /*

    :param pop: matrix of individuals as double (individuals x genes)
    :param individuals: number of individuals
    :param genes: number of genes
    :param fx: fitness function (double array x0 x1 ... xn)
    :param result: matrix of fitness values (individuals x 1)

    */

    
    if (fx_param.fx_method == Styblinski_Tang){
        for(int i = 0; i< gene_pool.individuals; i++){
              gene_pool.pop_result_set[i] = Styblinski_Tang_fx(gene_pool.pop_param_double[i], gene_pool.genes);
        }
    }
}

double Styblinski_Tang_fx(double* parameter_set, int genes){
    double result = 0;
    for (int i = 0; i < genes; i++){
        result += pow(parameter_set[i], 4) - 16 * pow(parameter_set[i], 2) + 5 * parameter_set[i];
    }
    return result / 2;
}
