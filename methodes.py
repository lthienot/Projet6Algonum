import numpy as np
import math
import matplotlib.pyplot as pyp

def step_euler(y, t, h, f) :
    return y + h*f(y, t)
