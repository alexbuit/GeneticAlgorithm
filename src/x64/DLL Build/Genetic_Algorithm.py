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
    
    cfg = config_ga_s()
    cfg.selection_param.selection_method = 1

    
    rtime_param = runtime_param_s(genes=2, max_iterations=1000,
                                         individuals=100, elitism=2,
                                         convergence_threshold=1e-8,
                                         convergence_window=5)
    

    dll = ctypes.CDLL(r"C:\Users\vanei\source\repos\Genetic Algorithm - C Branch\src\x64\DLL Build\Genetic Algrotihm.dll")
    geneticalgorithm = dll.Genetic_Algorithm
    print(geneticalgorithm)
    
    geneticalgorithm.argtypes = [zz_config_ga_s, zz_runtime_param_s]  
    for _ in range(0, 10):
        result = geneticalgorithm(cfg.cType(), rtime_param.cType())

    print(result)