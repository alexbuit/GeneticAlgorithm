
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "stdint.h"

#include "Genetic_Algorithm.h"

#include "../Utility/process.h"

#include "../Utility/pop.h"
#include "../Utility/crossover.h"
#include "../Utility/mutation.h"

#include "../Helper/Helper.h"

int main(){
    Genetic_Algorithm();
    return 0;
}

void Genetic_Algorithm(){
    struct gene_pool_s gene_pool;
    struct config_ga_s config_ga;

    seed_intXOR32();

    // struct flatten_param_s {
    // int flatten_method;
    // double flatten_factor;
    // double flatten_bias;
    // int flatten_optim_mode;
    // };

    // struct crossover_param_s {
    // int crossover_method;
    // double crossover_prob;
    // };

    // struct mutation_param_s {
    // int mutation_method;
    // double mutation_prob;
    // double mutation_rate;
    // };

    // struct fx_param_s {
    // int fx_method;
    // int fx_optim_mode;
    // };

    struct flatten_param_s flatten_param;
    flatten_param.flatten_method = 0;
    flatten_param.flatten_factor = 10.0f;
    flatten_param.flatten_bias = 1.0f;
    flatten_param.flatten_optim_mode = 0;

    struct crossover_param_s crossover_param;
    crossover_param.crossover_method = 0;
    crossover_param.crossover_prob = 0.5f;

    struct mutation_param_s mutation_param;
    mutation_param.mutation_method = 0;
    mutation_param.mutation_prob = 0.5f;
    mutation_param.mutation_rate = 0.1f;

    struct fx_param_s fx_param;
    fx_param.fx_method = 0;
    fx_param.fx_optim_mode = 0;

    struct selection_param_s selection_param;
    selection_param.selection_method = 0; 
    selection_param.selection_div_param = 0.0f;
    selection_param.selection_prob_param = 0.0f;
    selection_param.selection_temp_param = 0.0f;
    selection_param.selection_tournament_size = 0;

    struct config_ga_s config_ga;
    config_ga.selection_param = selection_param;
    config_ga.flatten_param = flatten_param;
    config_ga.crossover_param = crossover_param;
    config_ga.mutation_param = mutation_param;
    config_ga.fx_param = fx_param;

    

    // double loc = 0.0f;
    // double scale = 2.0f;
    // double factor = 10.0f;
    // double bias = 1.0f;

    int max_iterations = 1000;

    gene_pool.genes = 16;
    gene_pool.individuals = 32;
    gene_pool.elitism = 2;
    
    init_gene_pool(gene_pool);
    

    // While < iterations
    for (int i = 0; i < max_iterations; i++){

        // Process Population
        process_pop(gene_pool, config_ga, gene_pool.pop_result_set);
    }
    // Free pop_parameter_bin
    free_gene_pool(gene_pool);
    
}