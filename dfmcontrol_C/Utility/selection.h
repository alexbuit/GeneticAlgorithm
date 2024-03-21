

// Selection functions
static const int sel_roulette = 0;
static const int sel_rank_tournament = 1;
static const int sel_rank = 2;
static const int sel_rank_space = 3;
static const int sel_boltzmann = 4;


// gen purpose

void process_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected);



// Selection functions
void roulette_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected);
void rank_tournament_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected);
void rank_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected);
void rank_space_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected);
void boltzmann_selection(struct gene_pool_s gene_pool, struct selection_param_s selection_param, int* selected);

