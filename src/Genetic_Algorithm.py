import ctypes 

from Structs import *
# class ConfigGA(ctypes.Structure):
#     _fields_ = [
#         ("selection_param", SelectionParam),
#         ("flatten_param", FlattenParam),
#         ("crossover_param", CrossoverParam),
#         ("mutation_param", MutationParam),
#         ("fx_param", FxParam)
#     ]
if __name__ == "__main__":
    # $(OutDir)$(TargetName)$(TargetExt)
    
    cfg = ConfigGA()
    selection_param = create_selection_param(
        selection_method=0,
        selection_div_param=0,
        selection_prob_param=0,
        selection_temp_param=0,
        selection_tournament_size=0
        )
    cfg.selection_param = selection_param
    
    flatten_param = create_flatten_param(
        flatten_method=0,
        flatten_bias=0.0,
        flatten_factor=1.0,
        flatten_optim_mode=5)
    cfg.flatten_param = flatten_param
    
    crossover_param = create_crossover_param(
        crossover_method=6,
        crossover_prob=0.5
        )
    cfg.crossover_param = crossover_param
    
    mutation_param = create_mutation_param(
        mutation_method=0,
        mutation_prob=0.5,
        mutation_rate=6
        )
    cfg.mutation_param = mutation_param
    
    fx_param = create_fx_param(
        fx_method=1,
        fx_optim_mode=0,
        bin2double_factor=5,
        bin2double_bias=0
        )
    cfg.fx_param = fx_param
    
    runtime_param = create_runtime_param(genes=2, max_iterations=1000,
                                         individuals=100, elitism=2,
                                         convergence_threshold=1e-8,
                                         convergence_window=5)
    

    dll = ctypes.CDLL(r"C:\Users\vanei\source\repos\Genetic Algorithm - C Branch\src\x64\DLL Build\Genetic Algrotihm.dll")
    geneticalgorithm = dll.Genetic_Algorithm
    print(geneticalgorithm)
    
    geneticalgorithm.argtypes = [ConfigGA, RuntimeParam]  
    for _ in range(0, 10):
        result = geneticalgorithm(cfg, runtime_param)

    print(result)