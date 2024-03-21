#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.h"
#include "selection.h"




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


void process_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected){

        if(selection_param.selection_method ==sel_roulette){
        roulette_selection(gene_pool, selection_param, selected);
        }
        else if(selection_param.selection_method==sel_rank_tournament){
        rank_tournament_selection(gene_pool, selection_param, selected);
        }
        else if(selection_param.selection_method==sel_rank){
        rank_selection(gene_pool, selection_param, selected);
        }
        else if(selection_param.selection_method==sel_rank_space){
        rank_space_selection(gene_pool, selection_param, selected);
        }
        else if(selection_param.selection_method==sel_boltzmann){
        boltzmann_selection(gene_pool, selection_param, selected);
        }
        else{
        printf("Error: selection_method is not 0, 1, 2, 3 or 4\n");
        }

}

// Selection functions
void roulette_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param){
        /*

        :param pop: matrix of normalised fitness values for the population (individuals x 1)
        :param individuals: number of individuals
        :param genes: number of genes
        :param selected: matrix of the indices of selected individuals (individuals x 2)

        */

        // select individuals
        roulette_wheel(gene_pool.pop_result_set, gene_pool.individuals, gene_pool.individuals-gene_pool.elitism, gene_pool.selected_indexes);
        

}

void rank_tournament_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected){
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
void rank_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected){
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
void rank_space_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected){
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
void boltzmann_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected){
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
