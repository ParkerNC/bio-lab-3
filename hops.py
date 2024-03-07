import numpy as np
import json
import sys
import os

# activation function from class handout
def sign(x: int) -> int:
    if x >= 0:
        return 1
    else:
        return -1

def empty_vec(n: int, p: int) -> np.ndarray:
    #made on now I have to make them all
    return np.zeros((n, p))

def bipolar_vec(n: int, p: int) -> np.ndarray:
    #probably overkill but I am not getting rid of it now
    return np.random.choice([-1,1], (p, n))

def imprint_patterns(patterns: np.ndarray) -> np.ndarray:
    """
    imprint all patterns passed to function
    return the imprinted network
    patterns - numpy array of patterns to imprint.
    """
    n = patterns.shape[1]
    network_weights = empty_vec(n, n)

    #summation accross patterns
    for i in range(network_weights.shape[0]):
        for j in range(i):
            if i == j:
                #set diagonal to 0
                network_weights[i][j] = 0

            else:
                weight = 0

                for k in range(patterns.shape[0]):
                    #make weight value x_i^k * k_j^k
                    weight += patterns[k][i] * patterns[k][j]

                # 1/n
                # since weight vector is symmetric can be done for both sides
                network_weights[i][j] = weight / n
                network_weights[j][i] = weight / n


    return network_weights

def run(patterns: np.ndarray, weights: np.ndarray):
    count_imprinted = 0
    count_stable = 0
    for pattern in patterns:
        current_state = pattern.copy()
        new_state = pattern.copy()

        for i in range(weights.shape[0]):
            pattern_sum = 0
            count_imprinted += 1
            
            for j in range(weights.shape[1]):
                #h_i = w_ij * s_j
                pattern_sum += weights[i][j] * new_state[j]

            #s'
            new_state[i] = sign(pattern_sum)
        
        if np.array_equal(current_state, new_state): count_stable +=1

    return count_stable, count_imprinted

def run_expiriement():
    #make a pattern vector
    patterns = bipolar_vec(50, 100)
    stable_patterns = []
    imprinted_patterns = []

    for pattern in range(50):
        #get the first p desired patterns
        first_p_patterns = patterns[:pattern+1]
        #imprint first p patterns
        imprinted_weights = imprint_patterns(first_p_patterns)
        #run the network
        stable, imprinted = run(first_p_patterns, imprinted_weights)
        stable_patterns.append(stable)
        imprinted_patterns.append(imprinted)
    
    return stable_patterns, imprinted_patterns

if __name__ == "__main__":
    experiments = {}
    experiments["experiments"] = []
    for i in range(5):
        patterns, imprints = run_expiriement()
        experiment = {}
        experiment["imprints"] = []
        experiment["imprints"].append(imprints)
        experiment["stables"] = []
        experiment["stables"].append(patterns)

        experiments["experiments"].append(experiment)

    with open("datafile.json", "w") as out_file:
        out_file.write(json.dumps(experiments))
    