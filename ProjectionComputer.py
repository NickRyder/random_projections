import numpy as np
import numpy.random as npr
import helper as h

#See notes in OptimalRandomProjections for general use instructions



'''
This library will often take the estimator tuple:

estimator tuples consist of (randAx, determAx, randEstCount)
    rand_Ax -       n by n matrix so random estimators are (rand_Ax x)^T R  (R random iid gaussian)
    determ_Ax -     k-j by n matrix so our determistic estimators for our x space are (determ_Ax x)^T
    est_N  -        number of random estimators

'''


#7/31 - Cleaned up the package, wrote test commands
#7/31 - Made it take in both estimators at once to make projections correlated



'''
Inputs:
estimator_X - estimator tuple, see documentation above
estimator_W - estimator tuple, see documentation above
Outputs: 
projection_X - projection matrix for X space
projection_W - projection matrix for W space
'''
def compute_projection_matrix(estimator_X, estimator_W, is_diagonal=False):
    #unpack estimatorTuple
    (rand_Ax, determ_Ax, est_N) = estimator_X
    (rand_Aw, determ_Aw, est_N) = estimator_W


    #find ambient dimension
    n = rand_Ax.shape[0]

    # Empty array with n columns, need the columns to concatenate arrays later
    rand_X_component = np.array([], dtype=np.int64).reshape(0, n)
    rand_W_component = np.array([], dtype=np.int64).reshape(0, n)

    if est_N > 0:
        print("rand estimator...")
        rand_estimator = create_rand_estimator(est_N, n, True) # compute iid gaussian estimator
        if is_diagonal:
            rand_Ax_diag = np.diag(rand_Ax)
            rand_Aw_diag = np.diag(rand_Aw)
            rand_X_component = (1/np.sqrt(est_N)) * rand_estimator * rand_Ax_diag # scale random estimator
            rand_W_component = (1/np.sqrt(est_N)) * rand_estimator * rand_Aw_diag # scale random estimator
        else:
            rand_X_component = (1/np.sqrt(est_N)) * rand_estimator @ rand_Ax # scale random estimator
            rand_W_component = (1/np.sqrt(est_N)) * rand_estimator @ rand_Aw # scale random estimator

    return np.vstack([determ_Ax, rand_X_component]), np.vstack([determ_Aw, rand_W_component])


def create_rand_estimator(N, n, binary = False):
    if binary:
        int_N = np.int(np.ceil(N*n / 64))
        up = 2 ** 64
        list_unpack = np.unpackbits(npr.randint(up, size=int_N, dtype=np.uint64).view(np.uint8))[:N*n].reshape((N,n))# [:N*n]
        list_shift = 2*list_unpack.astype(np.int8) - 1
        return list_shift
    else:
        return npr.normal(size=(N, n))


'''
Inputs:
data_X -            data[i] is the ith data point
estimator_X -       estimator tuple, see documentation above
data_W -            data[i] is the ith data point
estimator_W -       estimator tuple, see documentation above
realizations_N -    number of random realizations
Outputs:
output_X -          output_X[i,j] is the ith data point projected under the jth realization
output_W -          output_W[i,j] is the ith data point projected under the jth realization
'''
def create_random_projection_realizations(data_X, estimator_X, data_W, estimator_W, realizations_N,is_diagonal = False):
    try:
        samples_N = data_X.shape[0]      # number of data points
        ambient_dim = data_X.shape[1]
        if data_X.shape != (samples_N, ambient_dim) or data_W.shape != (samples_N, ambient_dim):
            raise ValueError
        if len(estimator_X) != 3 or len(estimator_W) != 3 or estimator_X[0].shape != (ambient_dim,ambient_dim) or estimator_W[0].shape != (ambient_dim, ambient_dim):
            raise ValueError
    except ValueError:
        print("data not formatted properly")

    target_dim = estimator_X[-1] + estimator_X[1].shape[0]

    random_estimator_N = estimator_X[-1]

    #If deterministic, only sample once:

    if random_estimator_N > 0:
        output_X = np.empty((samples_N, realizations_N, target_dim))
        output_W = np.empty((samples_N, realizations_N, target_dim))
        for i in range(realizations_N):
            print(str(i))
            output_X[:, i, :], output_W[:, i, :] = apply_random_projection(data_X, estimator_X, data_W, estimator_W,is_diagonal)
    else:
        output_X = np.empty((samples_N, realizations_N, target_dim))
        output_W = np.empty((samples_N, realizations_N, target_dim))
        fixed_output = apply_random_projection(data_W, estimator_X, data_W, estimator_W,is_diagonal)
        for i in range(realizations_N):
            print(str(i))
            output_X[:, i, :], output_W[:, i, :] = fixed_output
    return output_X, output_W




'''
Inputs:
data_X -            data[i] is the ith data point
estimator_X -       estimator tuple, see documentation above
data_W -            data[i] is the ith data point
estimator_W -       estimator tuple, see documentation above
realizations_N -    number of random realizations
Outputs:
output_X -          output_X[i,j] is the ith data point projected under the jth realization
output_W -          output_W[i,j] is the ith data point projected under the jth realization
Outputs:
projected_data_X -  data_X projected under a realized random projection
projected_data_W -  data_W projected under a realized random projection
'''
def apply_random_projection(data_X, estimator_X, data_W, estimator_W,is_diagonal):
    print("computing projection matrix...")
    rand_projection_X, rand_projection_W = compute_projection_matrix(estimator_X, estimator_W,is_diagonal)

    print("applying projections...")
    output_t_X = rand_projection_X @ np.transpose(data_X)
    output_t_W = rand_projection_W @ np.transpose(data_W)

    return np.transpose(output_t_X), np.transpose(output_t_W)




'''
Takes in data, realizations, and estimators and computes the average dot product versus theoretical
'''
def verify_realizations(data_X, data_W, realizations_X, realizations_W, estimator_X, estimator_W):
    expected_dot = 0
    realizations_N = realizations_X.shape[1]

    for i in range(realizations_N):
        expected_dot += realizations_X[:, i] @ np.transpose(realizations_W[:, i])
    expected_dot = expected_dot / realizations_N
    theor_dot =  data_X @ np.transpose(estimator_X[0]) @ estimator_W[0] @ np.transpose(data_W)
    theor_dot += data_X @ np.transpose(estimator_X[1]) @ estimator_W[1] @ np.transpose(data_W)
    return np.amax(expected_dot - theor_dot)



'''
Generates random vectors x, w, random estimator tuples, realizes them 1000 times and then compares the average dot product
to the expected dot product.
'''
def test_create_random_projection_realizations():
    n = 10
    k = 4
    est_N = 1
    x = npr.rand(n)
    w = npr.rand(n)
    data_X = np.array([x])
    data_W = np.array([w])


    determ_X = npr.rand(k - est_N, n)
    determ_W = npr.rand(k - est_N, n)

    # determ_X = np.zeros((k-est_N,n))
    # determ_W = np.zeros((k-est_N,n))

    estimator_X = (npr.rand(n, n), determ_X, est_N)
    estimator_W = (npr.rand(n, n), determ_W, est_N)

    realizations_N = 10000
    realizations_X, realizations_W = create_random_projection_realizations(data_X, estimator_X, data_W, estimator_W, realizations_N)

    verify_realizations(data_X, data_W, realizations_X, realizations_W, estimator_X, estimator_W)





# def test_compute_projection_matrix():
#     import OptimalRandomProjections as orp
#     # n = 5
#     # k = 3
#     # lamb = .1
#     #
#     # d = np.array(range(n)) + 1
#     # d = np.flip(d, 0)
#     # cov_X = np.diag(d)
#     # cov_W = np.diag(d)
#     #
#     # SVDtuple = orp.preprocess_optimal_mixed_estimators(cov_X, cov_W)
#     # estimator_X, estimator_W, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(SVDtuple, k, lamb)
#
#     #n = 3, target_dim = 2, est_n = 1
#     n=3
#     estimator_X = np.diag([1, 0, 1]), np.array([0, 1, 0]), 2
#     estimator_W = estimator_X
#
#     x = npr.rand(n)
#     w = npr.rand(n)
#
#     N = 100000
#     sum = np.empty((3, 3))
#     for i in range(N):
#         est_X, est_W = compute_projection_matrix(estimator_X, estimator_W)
#
#         sum += est_X @ np.transpose(est_W)
#     print(sum/N)




#
#
# #Made as a sanity check to see that the math is in line with emperical observations
#
#
#
# #Takes as input x,w
# #Returns estimate of dot product (x,r)(x,w)
# def gaussianEstimate(x,w):
#     n = len(x)
#     r = npr.randn(n)
#     return np.dot(x,r)*np.dot(w,r)
#
#
# #Grabs random multivariate gaussian, mean zero
# #The covariance matrix has diagonals diag^2, and zeroes off diagonal
# def drawFromDiagonalDistribution(diag):
#     multivariateGauss = [npr.normal(scale=var) for var in diag]
#     return np.array(multivariateGauss)
#
# def drawFromMultivariateGaussian(covariance):
#     n = covariance.shape[0]
#     return npr.multivariate_normal(np.zeroes(n),covariance)
#
#
# def empericalVariance(x,w,Ax,Aw,N):
#     scaledX = Ax.dot(x)
#     scaledW = Aw.dot(w)
#
#     empericalVariance = 0
#     for i in range(N):
#         empericalVariance += np.power(gaussianEstimate(scaledW,scaledX) - scaledX.dot(scaledW),2)
#
#     return empericalVariance/N
#
#
# #Tools for studying expected bias and variance for a diagonal multivariate Gaussian distribution
#
# def empericalCovariance(d,N):
#     n = len(d)
#     empericalCovariance = np.zeros([n,n])
#     for i in range(N):
#         x = np.array(drawFromDiagonalDistribution(d))
#         empericalCovariance += np.outer(x,x)
#
#     return (empericalCovariance/(N-1))
#
# def empericalExpectedBias(d,Ax,Aw,N):
#     empericalExpectedBias = 0
#     for i in range(N):
#         empericalExpectedBias += empericalBiasSquared(drawFromDiagonalDistribution(d),drawFromDiagonalDistribution(d),Ax,Aw,N)
#     return empericalExpectedBias/N
#
# def empericalExpectedVariance(d,Ax,Aw,N):
#     empericalExpectedVariance = 0
#     for i in range(N):
#         empericalExpectedVariance += empericalVariance(drawFromDiagonalDistribution(d),drawFromDiagonalDistribution(d),Ax,Aw,N)
#     return empericalExpectedVariance/N
#
#
# if __name__ == '__main__':
#     #Set numpy settings
#     np.set_printoptions(linewidth=1000)
#
#
#     N = 10000
#     #Set testing variables
#     n = 10
#     d = npr.rand(n)
#     x = npr.rand(n)
#     w = npr.rand(n)
#     a = npr.rand(n)
#     Ax = np.diag(a)
#     Aw = np.diag(a)
#     scaledX = Ax.dot(x)
#     scaledW = Aw.dot(w)
#
#     #Test formulas for expected bias and expected variance
#     print(empericalExpectedBias(d, Ax, Aw, N) + empericalExpectedVariance(d, Ax, Aw, N))
#     print(np.sum((d*d*(a*a-1))**2) + np.sum((d*a)**4) + np.sum((d*a)**2)**2)
#
#     #Test formulas for bias squared and variance
#     # print(empericalBiasSquared(x,w,Ax,Aw,N))
#     # print((scaledX.dot(scaledW) - x.dot(w))**2)
#     # print(empericalVariance(x,w,Ax,Aw,N))
#     # print(scaledX.dot(scaledW)**2 + np.linalg.norm(scaledX)**2 * np.linalg.norm(scaledW)**2 )
#

