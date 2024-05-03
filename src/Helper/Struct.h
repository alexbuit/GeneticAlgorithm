
#ifndef STRUCT_H
#define STRUCT_H

struct gene_pool_s {
	int** pop_param_bin;
	int** pop_param_bin_cross_buffer;
	double** pop_param_double;
	double* pop_result_set;
	double* flatten_result_set;
	int* selected_indexes;
	int* sorted_indexes;
	int genes;
	int individuals;
	int elitism;
	int iteration_number;
};

struct selection_param_s {
	int selection_method;
	double selection_div_param;
	double selection_prob_param;
	double selection_temp_param;
	int selection_tournament_size;
};

struct flatten_param_s {
	int flatten_method;
	double flatten_factor;
	double flatten_bias;
	int flatten_optim_mode;
};

struct crossover_param_s {
	int crossover_method;
	double crossover_prob;
};

struct mutation_param_s {
	int mutation_method;
	double mutation_prob;
	int mutation_rate;
};

struct fx_param_s {
	int fx_method;
	int fx_optim_mode;
	double bin2double_factor;
	double bin2double_bias;
};

struct config_ga_s {
	struct selection_param_s selection_param;
	struct flatten_param_s flatten_param;
	struct crossover_param_s crossover_param;
	struct mutation_param_s mutation_param;
	struct fx_param_s fx_param;
};

struct runtime_param_s {
	int max_iterations;
	double convergence_threshold;
	int convergence_window;
	int genes;
	int individuals;
	int elitism;
};	

typedef struct gene_pool_s gene_pool_t;
typedef struct selection_param_s selection_param_t;
typedef struct flatten_param_s flatten_param_t;
typedef struct crossover_param_s crossover_param_t;
typedef struct mutation_param_s mutation_param_t;
typedef struct fx_param_s fx_param_t;
typedef struct config_ga_s config_ga_t;
typedef struct runtime_param_s runtime_param_t;


#endif // STRUCT_H