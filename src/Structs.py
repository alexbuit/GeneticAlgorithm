import ctypes
from re import S

class SelectionParam(ctypes.Structure):
    _fields_ = [
        ("selection_method", ctypes.c_int),
        ("selection_div_param", ctypes.c_double),
        ("selection_prob_param", ctypes.c_double),
        ("selection_temp_param", ctypes.c_double),
        ("selection_tournament_size", ctypes.c_int)
    ]

class FlattenParam(ctypes.Structure):
    _fields_ = [
        ("flatten_method", ctypes.c_int),
        ("flatten_factor", ctypes.c_double),
        ("flatten_bias", ctypes.c_double),
        ("flatten_optim_mode", ctypes.c_int)
    ]

class CrossoverParam(ctypes.Structure):
    _fields_ = [
        ("crossover_method", ctypes.c_int),
        ("crossover_prob", ctypes.c_double)
    ]

class MutationParam(ctypes.Structure):
    _fields_ = [
        ("mutation_method", ctypes.c_int),
        ("mutation_prob", ctypes.c_double),
        ("mutation_rate", ctypes.c_int)
    ]

class FxParam(ctypes.Structure):
    _fields_ = [
        ("fx_method", ctypes.c_int),
        ("fx_optim_mode", ctypes.c_int),
        ("bin2double_factor", ctypes.c_double),
        ("bin2double_bias", ctypes.c_double)
    ]

class ConfigGA(ctypes.Structure):
    _fields_ = [
        ("selection_param", SelectionParam),
        ("flatten_param", FlattenParam),
        ("crossover_param", CrossoverParam),
        ("mutation_param", MutationParam),
        ("fx_param", FxParam)
    ]

class RuntimeParam(ctypes.Structure):
    _fields_ = [
        ("max_iterations", ctypes.c_int),
        ("convergence_threshold", ctypes.c_double),
        ("convergence_window", ctypes.c_int),
        ("genes", ctypes.c_int),
        ("individuals", ctypes.c_int),
        ("elitism", ctypes.c_int),
    ]
    
def create_selection_param(selection_method: int = 0, selection_div_param: float = 0.0,
                           selection_prob_param: float = 0.0, selection_temp_param: float = 0.0,
                           selection_tournament_size: int = 0) -> SelectionParam:
    return SelectionParam(ctypes.c_int(selection_method), ctypes.c_double(selection_div_param),
                          ctypes.c_double(selection_prob_param), ctypes.c_double(selection_temp_param),
                          ctypes.c_int(selection_tournament_size))

def create_flatten_param(flatten_method: int = 0, flatten_factor: float = 0.0,
                         flatten_bias: float = 0.0, flatten_optim_mode: int = 0) -> FlattenParam:
    return FlattenParam(ctypes.c_int(flatten_method), ctypes.c_double(flatten_factor),
                        ctypes.c_double(flatten_bias), ctypes.c_int(flatten_optim_mode))

def create_crossover_param(crossover_method: int = 0, crossover_prob: float = 0.0) -> CrossoverParam:
    return CrossoverParam(ctypes.c_int(crossover_method), ctypes.c_double(crossover_prob))

def create_mutation_param(mutation_method: int = 0, mutation_prob: float = 0.0, mutation_rate: int = 0) -> MutationParam:
    return MutationParam(ctypes.c_int(mutation_method), ctypes.c_double(mutation_prob), ctypes.c_int(mutation_rate))

def create_fx_param(fx_method: int = 0, fx_optim_mode: int = 0, bin2double_factor: float = 0.0,
                    bin2double_bias: float = 0.0) -> FxParam:
    return FxParam(ctypes.c_int(fx_method), ctypes.c_int(fx_optim_mode),
                   ctypes.c_double(bin2double_factor), ctypes.c_double(bin2double_bias))

def create_runtime_param(max_iterations: int = 0, convergence_threshold: float = 0.0,
                         convergence_window: int = 0, genes: int = 0,
                         individuals: int = 0, elitism: int = 0) -> RuntimeParam:
    return RuntimeParam(ctypes.c_int(max_iterations), ctypes.c_double(convergence_threshold),
                        ctypes.c_int(convergence_window), ctypes.c_int(genes),
                        ctypes.c_int(individuals), ctypes.c_int(elitism))


class crossover_param_t:
    
    def __init__(self, method, prob):
        self.method = method
        self.prob = prob
        
    def to_c_type(self):
        res = ctypes.Structure(field=[(self.method, ctypes.c_int), (self.prob, ctypes.c_double)])
        return res