
from .t_functions import *
from .minima import *

__all__ = ["tfx", "wheelers_ridge", "booths_function", "michealewicz", "ackley", "Styblinski_Tang", "tfx_decorator",
           "mintfx", "minwheelers_ridge", "minbooths_function", "minmichealewicz", "minackley", "minStyblinski_Tang",
           "minackleyloc", "minbooths_functionloc", "minmichealewiczloc", "minStyblinski_Tangloc", "minwheelers_ridgeloc"]

# tfx = tfx_decorator(tfx)
# wheelers_ridge = tfx_decorator(wheelers_ridge)
# booths_function = tfx_decorator(booths_function)
# michealewicz = tfx_decorator(michealewicz)
# ackley = tfx_decorator(ackley)
# Styblinski_Tang = tfx_decorator(Styblinski_Tang)

allfx = [tfx, wheelers_ridge, booths_function, michealewicz, ackley, Styblinski_Tang]
dim2fx = [wheelers_ridge, booths_function]
ndimfx = [michealewicz, ackley, Styblinski_Tang]