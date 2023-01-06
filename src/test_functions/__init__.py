
from .t_functions import *

__all__ = ["tfx", "wheelers_ridge", "booths_function", "michealewicz", "ackley", "Styblinski_Tang"]

# tfx = tfx_decorator(tfx)
# wheelers_ridge = tfx_decorator(wheelers_ridge)
# booths_function = tfx_decorator(booths_function)
# michealewicz = tfx_decorator(michealewicz)
# ackley = tfx_decorator(ackley)
# Styblinski_Tang = tfx_decorator(Styblinski_Tang)

allfx = [tfx, wheelers_ridge, booths_function, michealewicz, ackley, Styblinski_Tang]
dim2fx = [wheelers_ridge, booths_function]
ndimfx = [michealewicz, ackley, Styblinski_Tang]