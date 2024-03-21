

static const int flat_linear = 0;
static const int flat_exponential  = 1;
static const int flat_logarithmic  = 2;
static const int flat_normalized = 3;
static const int flat_sigmoid = 4;
static const int flat_none = 5;

void process_flatten(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);

// Flattening functions
void lin_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);
void exp_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);
void log_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);
void norm_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);
void sig_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);
void no_flattening(struct gene_pool_s gene_pool, struct flatten_param_s flatten_param, int* result);


