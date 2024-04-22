
#ifndef SELECTION_H
#define SELECTION_H

#include "../Helper/Struct.h"


// Selection functions
static const int sel_roulette = 0;
static const int sel_rank_tournament = 1;
static const int sel_rank = 2;
static const int sel_rank_space = 3;
static const int sel_boltzmann = 4;


// gen purpose

void process_selection(gene_pool_t* gene_pool, selection_param_t* selection_param);

#endif

// Selection functions
// void roulette_selection(gene_pool_t *gene_pool, selection_param_t *selection_param);
// void rank_tournament_selection(gene_pool_t *gene_pool, selection_param_t *selection_param);
// void rank_selection(gene_pool_t *gene_pool, selection_param_t *selection_param);
// void rank_space_selection(gene_pool_t *gene_pool, selection_param_t *selection_param);
// void boltzmann_selection(gene_pool_t *gene_pool, selection_param_t *selection_param);

