
from dataclasses import dataclass
import json
import numpy as np
 

@dataclass
class paramset:

    conifg_dict: dict = json.load(r"c:\temp\config.json")

    def __init__(self):
        pass
    
    def __str__(self):
        return "paramset"
    
    def readbin(path:str):
        pass
    
if __name__ == "__main__":
    # print(paramset.conifg_dict)

    
