class gene_pool_s(ctypes.Structure):
    _fields_ = [
        ("genes", ctypes.c_int),
        ("individuals", ctypes.c_int),
        ("elitism", ctypes.c_int),
        ("iteration_number", ctypes.c_int),
    ]


class selection_param_s(ctypes.Structure):
    _fields_ = [
        ("selection_method", ctypes.c_int),
        ("selection_div_param", ctypes.c_double),
        ("selection_prob_param", ctypes.c_double),
        ("selection_temp_param", ctypes.c_double),
        ("selection_tournament_size", ctypes.c_int),
    ]


class flatten_param_s(ctypes.Structure):
    _fields_ = [
        ("flatten_method", ctypes.c_int),
        ("flatten_factor", ctypes.c_double),
        ("flatten_bias", ctypes.c_double),
        ("flatten_optim_mode", ctypes.c_int),
    ]


class crossover_param_s(ctypes.Structure):
    _fields_ = [
        ("crossover_method", ctypes.c_int),
        ("crossover_prob", ctypes.c_double),
    ]


class mutation_param_s(ctypes.Structure):
    _fields_ = [
        ("mutation_method", ctypes.c_int),
        ("mutation_prob", ctypes.c_double),
        ("mutation_rate", ctypes.c_int),
    ]


class fx_param_s(ctypes.Structure):
    _fields_ = [
        ("fx_method", ctypes.c_int),
        ("fx_optim_mode", ctypes.c_int),
        ("bin2double_factor", ctypes.c_double),
        ("bin2double_bias", ctypes.c_double),
    ]


class config_ga_s(ctypes.Structure):
    _fields_ = [
        ("selection_param", selection_param_s),
        ("flatten_param", flatten_param_s),
        ("crossover_param", crossover_param_s),
        ("mutation_param", mutation_param_s),
        ("fx_param", fx_param_s),
    ]


class runtime_param_s(ctypes.Structure):
    _fields_ = [
        ("max_iterations", ctypes.c_int),
        ("convergence_threshold", ctypes.c_double),
        ("convergence_window", ctypes.c_int),
        ("genes", ctypes.c_int),
        ("individuals", ctypes.c_int),
        ("elitism", ctypes.c_int),
    ]

