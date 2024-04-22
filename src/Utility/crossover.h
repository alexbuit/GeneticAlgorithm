
#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "../Helper/Struct.h"

static const int cross_single_point = 0;
static const int cross_two_point = 1;
static const int cross_uniform = 2;
static const int cross_complete = 3;
static const int cross_single_point32 = 4;
static const int cross_two_point32 = 5;
static const int cross_uniform32 = 6;
static const int cross_complete32 = 7;

void process_crossover(gene_pool_t* gene_pool, crossover_param_t* crossover_param);

#endif