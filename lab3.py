import numpy as np
import matplotlib.pyplot as plt
import sys

# Define the sign
def activate(x):
    if (x >= 0):
        return 1
    else:
        return -1

def imprint(patterns, pShape, n):
    """return imprinted weight vector

        Args:
            np.2darray (patterns): these are our patterns to imprint
            int (pShape): shape of patterns vector
            int (n): number of neurons

        Returns:
            np.2darray: vector of imprinted weights
    """
    weights = np.zeros(shape=(n,n))
    # 3d for loop to go through each weight, and each pattern for each weight
    for i in range(weights.shape[0]):
        for j in range(i):
            # diagonal to 0
            if i == j:
                weights[i][j] = 0
            else:
                sum = 0

                # calculate the weight by summing the different values in patterns
                for k in range(pShape[0]):
                    sum += patterns[k][i] * patterns[k][j]
                
                # this is to make the calculation more efficient
                # since the weight vector is symmetric
                weights[i][j] = sum / n
                weights[j][i] = sum / n

    return weights

def runNetwork(pPatterns, weights, wShape):
    numStable = 0
    numImprinted = 0
    for p in pPatterns:
        state = p.copy()
        newState = p.copy()

        for i in range(wShape[0]):
            sum = 0

            #calculate h
            for j in range(wShape[1]):
                sum += weights[i][j] * newState[j]
            #calculate s'
            newState[i] = activate(sum)

            numImprinted += 1

        # i check the two state vectors against each other instead of
        # each neuron state b/c i was being lazy
        if checkStability(state, newState):
            numStable += 1

    return newState, numStable, numImprinted

def checkStability(s, sPrime):
    return np.array_equal(s, sPrime)

def main(sysN, sysP):
    # make bipolar patterns
    numExperiments = 5
    numPatterns = sysP
    numNeurons = sysN
    numStable = np.zeros(shape=(numExperiments, numPatterns))
    numImpr = np.zeros(shape=(numExperiments, numPatterns))
        
    
    for i in range(numExperiments):
        patterns = np.random.choice([-1, 1], (numPatterns, numNeurons))
        
        # for each of the vectors patterns[0:p], imprint a weight vector,
        # calculate the number of stables by running each pattern through the network
        for p in range(patterns.shape[0]):
            vect = patterns[:p+1]
            imprinted = imprint(vect, vect.shape, vect.shape[1])
            result, currNumStable, currNumImpr = runNetwork(vect, imprinted, imprinted.shape)
            numStable[i][p] += currNumStable
            numImpr[i][p] += currNumImpr

        # just to make sure its working while executing
        print(f'Experiment {i}, {numNeurons} Neurons, {numPatterns} Patterns')
        print(''.join(['~' for x in range(50)]))
        print('numStable: ', numStable[i])
    
    # plot stable
    for i in range(numExperiments):
        plt.plot(np.arange(0, numPatterns), numStable[i], label=f'Experiment {i}')
    plt.xlabel('Number of patterns imprinted')
    plt.ylabel('Number of stable networks')
    plt.title('Number Stable vs Number Imprinted')
    plt.legend()
    plt.savefig(f'numStable{numNeurons}_{numPatterns}.png')

    # plot avg
    plt.clf()
    avg = np.mean(numStable, axis=0)
    std = np.std(numStable, axis=0)
    plt.plot(np.arange(0, numPatterns), avg)
    plt.fill_between(np.arange(0, numPatterns), np.subtract(avg, std), np.add(avg, std), alpha=0.2)
    plt.xlabel('Number of patterns imprinted')
    plt.ylabel('Number of stable networks')
    plt.title('Number Stable vs Number Imprinted')
    plt.legend()
    plt.savefig(f'avgStable{numNeurons}_{numPatterns}.png')

    #plot fraction instable
    plt.clf()
    for i in range(numExperiments):
        fractUnstable = numStable[i].copy()
        '''if numImpr[i] != 0:  '''
        fractUnstable /= np.arange(1, numPatterns+1)
        fractUnstable -= 1
        fractUnstable = np.abs(fractUnstable)
        plt.plot(np.arange(0, numPatterns), fractUnstable, label=f'Experiment {i}')
    plt.xlabel('Number of patterns imprinted')
    plt.ylabel('Fraction of unstable networks')
    plt.title('Average Fraction Unstable vs Number Imprinted')
    plt.legend()
    plt.savefig(f'fractUnstable{numNeurons}_{numPatterns}.png')

    # plot avg
    plt.clf()
    fractUnstable = np.zeros(shape=(numExperiments, numPatterns))
    for i in range(numExperiments):
        fractUnstable[i] = numStable[i].copy()
        '''if numImpr[i] != 0:  '''
        fractUnstable[i] /= np.arange(1, numPatterns+1)
        fractUnstable[i] -= 1
        fractUnstable[i] = np.abs(fractUnstable[i])
    avg = np.mean(fractUnstable, axis=0)
    std = np.std(fractUnstable, axis=0)
    plt.plot(np.arange(0, numPatterns), avg)
    plt.fill_between(np.arange(0, numPatterns), np.subtract(avg, std), np.add(avg, std), alpha=0.2)
    plt.xlabel('Number of patterns imprinted')
    plt.ylabel('Number of stable networks')
    plt.title('Average Fraction Unstable vs Number Imprinted')
    plt.legend()
    plt.savefig(f'avgUnstable{numNeurons}_{numPatterns}.png')


if __name__ == "__main__":
    # not error checking sys args bc im lazy and i made a script to run code :)
    if len(sys.argv) != 3:
        print("usage: lab3.py num_neurons num_patterns")
    else:
        main(int(sys.argv[1]), int(sys.argv[2]))