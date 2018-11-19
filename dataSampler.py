import numpy as np
import numpy.random as npr
import idx2numpy as idx2np

##Artificial Data

#Spits out a N x n matrix
#each row is a different data sample
def generateNSamples(covariance, N):
    n = covariance.shape[0]
    return np.array([npr.multivariate_normal(np.zeros(n),covariance) for i in range(N)])

def generateSamplesMixed(covariance_list, mean_list, distribution, samplesN):
    return np.array([_generateSampleMixed(covariance_list,mean_list,distribution) for i in range(samplesN)])


# Spits out a point from distribution 1 with probability p and distribution 2 with probability 1-p
def _generateSampleMixed(covariance_list, mean_list, distribution):
    n = len(distribution)
    gauss_index = npr.choice(np.arange(n), p=distribution)
    return npr.multivariate_normal(mean_list[gauss_index], covariance_list[gauss_index])

##Emperical Data
#Takes in file name and spits out data where data[i] is ith data point
def import_idx(filename):
    output = idx2np.convert_from_file(filename)
    output.flags.writeable = True # Allows manipulation of data
    return output

#Takes in in_data where in_data[i] is two dimensional and returns out_data where out_data[i] is in_data[i] flattened
def flatten_2D_data(data):
    (dataN, axis1N, axis2N) = data.shape
    return np.reshape(data,(dataN, axis1N*axis2N))

def random_sample(data, N):
    npr.shuffle(data)
    return data[:N]

def draw_MNIST(N):
    imported_data = import_idx("data/train-images-idx3-ubyte")
    flattened_data = flatten_2D_data(imported_data)
    sampled_data = random_sample(flattened_data,N)

    return sampled_data

def draw_cifar(N):
    import pickle
    with open("data/cifar_10_data_batch_1", 'rb') as fo:
        cifar = pickle.load(fo, encoding='bytes')
    cifar_data = cifar[b'data']
    return random_sample(cifar_data, N)

def draw_isolet():
    with open('data/isolet5.data', 'rt') as f:
        return np.loadtxt(f, delimiter=',')[:,:-1] # chops off the classifier

def import_repeat_consumption(dir):
    import scipy.sparse
    array_coo = np.loadtxt("data/repeat_consumption/" + dir + "/test.csv",delimiter=",", dtype=np.int)
    np_array_coo = scipy.sparse.coo_matrix((array_coo[:,2],(array_coo[:,0],array_coo[:,1])))
    return np_array_coo.toarray()

def import_arcene():
    import csv
    data = []
    with open('data/arcene/test.data', 'rt') as csvfile:
        data_lines = csv.reader(csvfile, delimiter=' ')
        for item in data_lines:
            item = np.array(item[:-1])
            data.append(item.astype(float))
        csvfile.close()
    return np.array(data)

def import_slice_localization():
    import csv
    data = []
    with open('data/slice_localization_data.csv', 'rt') as csvfile:
        data_lines = csv.reader(csvfile)
        first_line = True
        for item in data_lines:
            if not first_line:
                item = np.array(item[1:])
                data.append(item.astype(float))
            else:
                first_line = False
        csvfile.close()
    return np.array(data)


def import_libsvm(filename):
    from sklearn import datasets

    return datasets.load_svmlight_file('data/' + filename)