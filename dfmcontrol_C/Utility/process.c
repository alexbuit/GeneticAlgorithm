#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include "../Helper/Helper.h"
#include "flatten.h"
#include "../Function/Function.h"
#include "selection.h"
#include "crossover.h"

#include "process.h"
#include "mutation.h"

void process_pop(struct gene_pool_s gene_pool, struct config_ga_s config_ga, int* selected){
// TODO: check individual even nr 
// TODO: refractor individuals and genes to _count

        process_fx(gene_pool, config_ga.fx_param); // pop, individuals, genes -> struct ?

        double* result = malloc(gene_pool.individuals * sizeof(double));
        process_flatten(gene_pool, config_ga.flatten_param, result);

        indexed_merge_sort(gene_pool.pop_result_set, gene_pool.sorted_indexes, gene_pool.individuals);

        process_selection(gene_pool, config_ga.selection_param);
        free (result);

        // crossover
        process_crossover(gene_pool, config_ga.crossover_param);

        // mutation
        mutate32(gene_pool, config_ga.mutation_param);

}

