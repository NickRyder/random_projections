import numpy as np
import numpy.random as npr
import sys
import time
from scipy.stats import ortho_group

def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def randomPSD(n, d = None):
    if d is None:
        d = npr.rand(n)

    return random_orthogonal_conjugation(np.diag(d))



# randomly conjugates A using scipy
def random_orthogonal_conjugation(A):
    n = A.shape[0]
    orth = ortho_group.rvs(n)
    return orth @ A @ np.transpose(orth)


'''found from https://github.com/UCIDataLab/repeat-consumption/blob/master/util/io.py'''
def load_txt(filename, delimiter=',', verbose=True):
    if verbose:
        print('--> Loading ', filename, ' with np.loadtxt was ')
    sys.stdout.flush()
    t = time.time()
    d = np.loadtxt(filename, delimiter=delimiter)
    if verbose:
        print('%.3f s' % (time.time() - t))
    return d


'''takes in numpy matrix and returns boolean indicating whether it is diagonal. Off diagonal must be exact zero'''
def is_diagonal(matrix):
    return np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0


def psuedo_power(array, pow, thresh = 0): # np.power(10.0,-12)
    array_copy = np.copy(array.astype(np.float))
    for i in range(len(array)):
        if np.abs(array[i]) <= thresh:
            array_copy[i] = 0
        else:
            try:
                array_copy[i] = np.power(array[i], pow)
            except RuntimeWarning:
                array_copy[i] = 0
    return array_copy


'''Takes in data_regress, data_values, and spits out rows which correspond to 0, 1'''
def trim_classification(data_regress, data_values):
    samples_N = data_regress.shape[0]
    ambient_dim = data_regress.shape[1]
    output_regress = np.empty((0, ambient_dim))
    output_values = np.empty((0))

    for i in range(samples_N):
        if data_values[i] == 0 or data_values[i] == 1:
            output_regress = np.vstack((output_regress, data_regress[i]))
            output_values = np.append(output_values, data_values[i])

    return output_regress, output_values


