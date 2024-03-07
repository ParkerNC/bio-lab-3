import numpy as np
import sys
import os

def bipolar_vec(n, p):
    return np.random.choice([-1,1], (p, n))

if __name__ == "__main__":
    #patterns = np.array([bipolar_vec(100) for i in range(50)])
    patterns = bipolar_vec(100, 50)
    print(len(patterns), len(patterns[0]))