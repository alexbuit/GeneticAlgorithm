#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.h"
#include "flatten.h"
#include "../Function/Function.h"
#include "selection.h"

#include "process.h"

void process_pop(struct gene_pool_s gene_pool, struct config_ga_s config_ga, int* selected){
// TODO: check individual even nr 
// TODO: refractor individuals and genes to _count

        process_fx(gene_pool, config_ga.fx_param); // pop, individuals, genes -> struct ?

        double* result = malloc(gene_pool.individuals * sizeof(double));
        process_flatten(gene_pool, config_ga.flatten_param, result);

        sort_population_idx(gene_pool);

        process_selection(gene_pool, config_ga.selection_param);
        free (result);
        

        // crossover
        

        // sort
        sort_population_idx(gene_pool);

        // mutation

        // free
}

void sort_population_idx(struct gene_pool_s gene_pool){

        // make a copy of the population
        double* pop_result_set_copy = malloc(gene_pool.individuals * sizeof(double));
        for(int i = 0; i < gene_pool.individuals; i++){
                pop_result_set_copy[i] = gene_pool.pop_result_set[i];
                gene_pool.selected_indexes[i] = i;
        }

        // sort the population
        indexed_merge_sort(pop_result_set_copy, gene_pool.selected_indexes, gene_pool.individuals);
        
        // free
        free(pop_result_set_copy);
}
