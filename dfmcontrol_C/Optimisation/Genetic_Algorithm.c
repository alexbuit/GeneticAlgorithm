
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdint.h"

#include "Genetic_Algorithm.h"

#include "../Utility/process.h"

#include "../Utility/pop.h"
#include "../Utility/crossover.h"
#include "../Utility/mutation.h"
#include "../Utility/selection.h"
#include "../Utility/flatten.h"

#include "../Function/Function.h"

#include "../Helper/Helper.h"
#include "../Helper/Struct.h"


void Genetic_Algorithm(){
    gene_pool_t gene_pool;

    // seed_intXOR32();

    // flatten_param_t {
    // int flatten_method;
    // double flatten_factor;
    // double flatten_bias;
    // int flatten_optim_mode;
    // };

    // crossover_param_t {
    // int crossover_method;
    // double crossover_prob;
    // };

    // mutation_param_t {
    // int mutation_method;
    // double mutation_prob;
    // double mutation_rate;
    // };

    // fx_param_t {
    // int fx_method;
    // int fx_optim_mode;
    // };

    flatten_param_t flatten_param;
    flatten_param.flatten_method = 0;
    flatten_param.flatten_factor = 1.0f;
    flatten_param.flatten_bias = 0.0f;
    flatten_param.flatten_optim_mode = flat_none;

    crossover_param_t crossover_param;
    crossover_param.crossover_method = cross_uniform32;
    crossover_param.crossover_prob = 0.5f;

    mutation_param_t mutation_param;
    mutation_param.mutation_method = 0;
    mutation_param.mutation_prob = 0.5f;
    mutation_param.mutation_rate = 6;

    fx_param_t fx_param;
    fx_param.fx_method = fx_Wheelers_Ridge;
    fx_param.fx_optim_mode = 0;
    fx_param.bin2double_factor = 5.0f;
    fx_param.bin2double_bias = 0.0f;

    selection_param_t selection_param;
    selection_param.selection_method = 0; 
    selection_param.selection_div_param = 0.0f;
    selection_param.selection_prob_param = 0.0f;
    selection_param.selection_temp_param = 10.0f;
    selection_param.selection_tournament_size = 0;

    config_ga_t config_ga;
    config_ga.selection_param = selection_param;
    config_ga.flatten_param = flatten_param;
    config_ga.crossover_param = crossover_param;
    config_ga.mutation_param = mutation_param;
    config_ga.fx_param = fx_param;

    printf("%f", selection_param.selection_temp_param);

    // double loc = 0.0f;
    // double scale = 2.0f;
    // double factor = 10.0f;
    // double bias = 1.0f;

    int max_iterations = 1000000;

    gene_pool.genes = 2;
    gene_pool.individuals = 32;
    gene_pool.elitism = 2;
    
    init_gene_pool(&gene_pool);

    fill_pop(&gene_pool);

    double best_res = 0.0f;
    
    // // While < iterations
    for (int i = 0; i < max_iterations; i++){

        // Process Population
        process_pop(&gene_pool, &config_ga);
        if (gene_pool.pop_result_set[gene_pool.sorted_indexes[gene_pool.individuals-1]] - best_res > 0.0f){
            printf("Iteration: %d Gain: %0.3f best res: %0.3f (idx = %d), 2nd best res: %0.3f (idx = %d) 3rd best res %0.3f (idx = %d)\n", i,
            (gene_pool.pop_result_set[gene_pool.sorted_indexes[gene_pool.individuals-1]] - best_res),
            gene_pool.pop_result_set[gene_pool.sorted_indexes[gene_pool.individuals-1]], gene_pool.sorted_indexes[gene_pool.individuals-1],
            gene_pool.pop_result_set[gene_pool.sorted_indexes[gene_pool.individuals-2]], gene_pool.sorted_indexes[gene_pool.individuals-2],
            gene_pool.pop_result_set[gene_pool.sorted_indexes[gene_pool.individuals-3]], gene_pool.sorted_indexes[gene_pool.individuals-3]);
        }
        best_res = gene_pool.pop_result_set[gene_pool.sorted_indexes[gene_pool.individuals-1]];

    }
    // Free pop_parameter_bin
    free_gene_pool(&gene_pool);
    
}

int main(){
    for (int i = 0; i < 10; i++){
        printf("\n Run number: %d\n", i);
        Genetic_Algorithm();
    }
    return 0;
}