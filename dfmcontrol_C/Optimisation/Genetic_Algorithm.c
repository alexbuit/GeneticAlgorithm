
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

void Genetic_Algorithm(){
    struct gene_pool_s gene_pool;
    struct config_ga_s config_ga;

    seed_intXOR32();

    // double loc = 0.0f;
    // double scale = 2.0f;
    // double factor = 10.0f;
    // double bias = 1.0f;

    int max_iterations = 1000;

    gene_pool.genes = 16;
    gene_pool.individuals = 32;
    
    init_gene_pool(&gene_pool);
    

    // While < iterations
    for (int i = 0; i < max_iterations; i++){

        // Process Population
        process_pop(gene_pool, config_ga, gene_pool.pop_result_set);

        // Process crossover
        int** pop_parameter_bin_new = malloc(gene_pool.individuals * sizeof(int*));
        for (int i = 0; i < gene_pool.individuals; i++){
            pop_parameter_bin_new[i] = malloc(gene_pool.genes * sizeof(int));
        }
        
        crossover(pop_parameter_bin, individuals, genes, selected, pop_parameter_bin_new);

        // Process mutation
        mutation(pop_parameter_bin_new, individuals, genes);

    // Free pop_parameter_bin
    for (int i = 0; i < individuals; i++){
        free(pop_parameter_bin[i]);
    }
    free(pop_parameter_bin);

}