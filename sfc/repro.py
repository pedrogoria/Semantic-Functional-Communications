import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
