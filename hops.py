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

def rand_vec(int_range: int) -> np.ndarray:
    #takes a range and returns a randomized vector of elements in that range
    arr = np.arange(int_range)
    np.random.shuffle(arr)

    return arr

def empty_vec(n: int, p: int) -> np.ndarray:
    #made on now I have to make them all
    return np.zeros((n, p))

def bipolar_vec(n: int, p: int) -> np.ndarray:
    #probably overkill but I am not getting rid of it now
    return np.random.choice([-1,1], (p, n))

def estimate_basin(pattern: np.ndarray, weights: np.ndarray) -> int:
    #generate random permutation of 100
    rand_array = rand_vec(100)
    #current network pattern
    test_net = pattern.copy()
    #consider the first 50 elements of permute array
    for i in range(50):
        #flip every bit a given position
        test_net[rand_array[i]] = -(test_net[rand_array[i]])
        convergent_state = compute_state(weights, test_net)
        for j in range(9):
            convergent_state = compute_state(weights, convergent_state)

        if not np.array_equal(convergent_state, pattern):
            return i+1
    
    return 50


def imprint_patterns(patterns: np.ndarray) -> np.ndarray:
    """
    imprint all patterns passed to function
    return the imprinted network
    patterns - numpy array of patterns to imprint.
    """
    #n
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

def compute_state(weights: np.ndarray, state: np.ndarray) -> np.ndarray:
    ran_state = state.copy()
    for i in range(weights.shape[0]):
        pattern_sum = 0
            
        for j in range(weights.shape[1]):
            #h_i = w_ij * s_j
            pattern_sum += weights[i][j] * ran_state[j]

        #s'
        ran_state[i] = sign(pattern_sum)
    
    return ran_state

def run(patterns: np.ndarray, weights: np.ndarray) -> tuple[int, int, int]:
    count_imprinted = 0
    count_stable = 0
    basin_size = 0
    for pattern in patterns:
        current_state = pattern.copy()
        new_state = pattern.copy()
        new_state = compute_state(weights, new_state)
        count_imprinted += weights.shape[1]
        
        #check for stability
        if np.array_equal(current_state, new_state):
            #array is stable, need to do basin of attraction 
            count_stable +=1
            basin_size = estimate_basin(current_state, weights)
        else:
            #not stable, set basin of attraction to 0
            basin_size = 0

    return count_stable, count_imprinted, basin_size

def run_expiriement() -> tuple[int, int]:
    #make a pattern vector
    patterns = bipolar_vec(100, 50)
    stable_patterns = []
    imprinted_patterns = []
    basins = []

    for pattern in range(50):
        #get the first p desired patterns
        first_p_patterns = patterns[:pattern+1]
        #imprint first p patterns
        imprinted_weights = imprint_patterns(first_p_patterns)
        #run the network
        stable, imprinted, basin = run(first_p_patterns, imprinted_weights)
        stable_patterns.append(stable)
        imprinted_patterns.append(imprinted)
        basins.append(basin)
    
    return stable_patterns, imprinted_patterns

if __name__ == "__main__":
    #main file, run 5 experiments
    experiments = {}
    experiments["experiments"] = []
    for i in range(5):
        #get stable patterns and number of imprints per experiments
        patterns, imprints, basins = run_expiriement()
        print(patterns, imprints, basins)
        experiment = {}
        experiment[i] = []
        experiment[i]["imprints"] = []
        experiment[i]["imprints"].append(imprints)
        experiment[i]["stables"] = []
        experiment[i]["stables"].append(patterns)
        experiment[i]["basins"] = []
        experiment[i]["basins"].append(basins)

        experiments["experiments"].append(experiment)

    #throw everything in a json file for graphing later
    with open("datafile.json", "w") as out_file:
        out_file.write(json.dumps(experiments))
    