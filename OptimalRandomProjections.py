import numpy as np
import numpy.random as npr
import scipy.linalg as linal
import time
from NumericalProjectionSolver import *
import helper


#TODO: Using SVD to compute sqrt of symmetric matrix, probably not optimal




#7/18 - removed q and nu
#7/25 - cleaned up code greatly
#7/26 - added test commands



# General Use:
# Given two covariance matrices, call preprocessOptimalMixedEstimators to get a preprocess_SVD triple
# Then with a target dimension and lambda parameter, call optimalMixedEstimators()
# You will get out two estimatorTuples, one for the x space and one for the w space
# To get an actual realization of a projection, use ProjectionComputer's method computeProjectionMatrix
# To get a whole array of realizations use ProjectionComputer's createRandomProjectionRealization
# These arrays of realizations and estimatorTuples can be used with the dataAnalytics library for analysis



# Toy Example: Results on diagonal symmetric case
# covarianceX = covarianceW = D is a diagonal matrix
# preprocessOptimalMixedEstimators(D,D) = (D^{-1/2}, D^{-1/2}, D)
# calling then optimalMixedEstimators(preprocess, k 0) = (I, empty matrix, k)




#TODO: Not used in current workflow
# Finds optimal solution to:
# sum_i (a_i^2 - d_i)^2 + \lambda ((\sum_i a_i^4) + (\sum_i a_i^2)^2)
# Inputs:   d -     list or 1-d np array of positive real numbers (if pre-sorted, use internal command instead)
#           lamb -  positive scalar
# Outputs:  arg_min -       list of positive reals which achieve minimum for above quantity
#           cutoff_min -    index of first nonzero entry in a
#           obj_bias -      bias in our minimal objective: sum_i (a_i^2 - d_i)^2)^q
#           obj_var -       variance in our minimal objective: ((\sum_i a_i^4) + (\sum_i a_i^2)^2)
# Notes: Naive implementation: linear run time, linear memory
def mixed_objective_diagonal_solver(d, lamb):
    if lamb < 0:
        raise ValueError('non convex combination encountered: lambda < 0')

    #Cast d as a numpy arry
    d = np.array(d)

    #Sort d
    d_arg_sort = np.argsort(d)
    d_arg_sort_inv = helper.invert_permutation(d_arg_sort)
    d_sorted = [d[d_arg_sort[i]] for i in range(len(d))]

    #call internal sorted command
    arg_min_sorted, cutoff_min, obj_bias, obj_var = _mixed_objective_sorted_diagonal_solver(d_sorted, lamb)

    #unsort arguement
    arg_min = [arg_min_sorted[d_arg_sort_inv[i]] for i in range(len(d))]
    return arg_min, cutoff_min, obj_bias, obj_var


# Internal command which takes in a sorted list
# Created to avoid sorting a list multiple times when unnecessary
def _mixed_objective_sorted_diagonal_solver(d, lamb):
    #Check inputs
    if lamb < 0:
        raise ValueError('non convex combination encountered: lambda < 0')

    # Mathematical Observation:
    # The optimal $a$ will have the first $p$ coordinates zero,
    # rest equal to \sqrt{\frac{d_i (1+ \lambda + m \lambda) - \lambda \sum_S d_i}{(1+\lambda + m \lambda)(1+\lambda)}}
    # This optimal $p$ is the first such that d_i (1+ \lambda + m \lambda) - \lambda \sum_{i \geq p} d_i > 0

    # Naive Implementation: Linearly traverse until condition is met
    d = np.array(d)

    n = len(d)
    cutoff = 0 # number of zeroed out coords
    dSigma = sum(d)

    while cutoff is not n and d[cutoff] * (1 + lamb + (n - cutoff) * lamb) - lamb * dSigma < 0:
        dSigma -= d[cutoff]
        cutoff += 1

    arg_head = [0] * cutoff
    arg_tail = np.sqrt(
        (d[cutoff:] * (1 + lamb + (n - cutoff) * lamb) - lamb * dSigma) / ((1 + lamb + (n - cutoff) * lamb) * (1 + lamb)))

    arg = arg_head + arg_tail.tolist()
    arg_np = np.array(arg)
    obj_bias = np.sum(np.power(arg_np * arg_np - d, 2))
    obj_var = (sum(np.power(arg_np, 4)) + np.power(sum(arg_np * arg_np), 2))
    return arg, cutoff, obj_bias, obj_var


#TODO: Possible speed up: use binary search even though its not unimodular
# Finds optimal solution to:
# min_{est_N} (sum_i (a_i^2 - d_i)^2) + (\lambda / est_N) ((\sum_i a_i^4) + (\sum_i a_i^2)^2)
# where we re allowed to remove k - est_N of the d_i's from the optimization problem
# For j = k, we are allowed to remove any k of the d_i's, the objective is summed over the remaining (\sum_i d_i^2)^q
# Inputs:   d -         list or 1-d np array of positive real numbers sorted smallest to largest
#           lamb -      positive scalar for the objective, if negative returns all random estimator
#           k -         total number of estimators (must be between 1 and k-1)
#           thresh -    for lamb <= thresh we use objVariance as the obj instead for numerical stability
# Outputs:
#           arg_min -   list of positive reals which achieve minimum for above quantity
#           est_N_min - integer between 0 and k which indicates how many random estimators are ideal
#           cutoff -    number of zeroed out coordinates (not including the deterministic part)
#           determ -    the indices of the k - j coordinates for which we remove the d_i
#           obj -       minimal objective, unless lamb <= thresh, then we use just the objVariance
# Notes: Naive implementation: quadratic run time, linear memory
def mixed_estimator_diagonal_solver(d, lamb, k, thresh=0):

    # Convert data types
    d = np.array(d)
    n = len(d)

    if k <= 0 or k >= n:
        raise ValueError("k is not a feasible target dimension")

    # Handle unsorted d
    d_arg_sort = np.argsort(d)
    d_arg_sort_inv = helper.invert_permutation(d_arg_sort)
    d_sorted = np.array([d[d_arg_sort[i]] for i in range(len(d))])



    if lamb < 0:
        # raise ValueError('non convex combination encountered: lambda < 0')
        # Hardcode all random case
        print("Negative lambda encountered: Using all random projections")
        obj_bias = 0
        arg_min = np.sqrt(d)
        obj_var = (1.0/k)*(sum(np.power(arg_min, 4)) + np.power(sum(arg_min * arg_min), 2))

        arg_min_ordered, est_N_min, cutoff_min, determ_resorted, obj_bias_min, obj_var_min = arg_min, k, 0, [], obj_bias, obj_var
    # elif lamb == 0:
        # # Look over unbiased case, computes all variances, and then finds minimum among these
        #
        # # d_sorted = np.array([d[d_arg_sort[i]] for i in range(len(d))])
        # d_exp2 = d_sorted ** 2
        # d_exp2_cum_sum = np.cumsum(d_exp2) # Sum_i d_i^2
        #
        # d_cum_sum = np.cumsum(d_sorted) # Sum_i  d_i
        # d_cum_sum_exp2 = d_cum_sum ** 2 # (Sum_i d_i)^2
        #
        # var_before_scale = d_exp2_cum_sum + d_cum_sum_exp2
        # var_chop = var_before_scale[-k:]
        # scaling_floats = (np.reciprocal(np.arange(k, dtype=np.dtype(float)) + 1))
        # var_scaled = var_chop * scaling_floats
        #
        #
        # random_estimator_N = np.argmin(var_scaled) + 1 # number of random estimators
        # zero_coords_N = 0
        # determ_coords = np.arange(n - k + j, n)
        # min_obj = var_scaled[random_estimator_N]
        #
        #
        #
        # # Need to unscramble
        # # # Unsort arguement
        # # reSortedA = np.array([aMin[d_arg_sort_inv[i]] for i in range(len(aMin))])
        # #
        # # # Determine indices of the determinstic entries
        # # determ = range(n - k + jMin, n)
        # # determResorted = np.array([d_arg_sort[i] for i in determ])
        #
        # print(var_scaled)
        #
        #
        #
        #
        #
        # # hardcode all random case
        # reSortedA, jMin, pMin, determResorted, objMin = np.sqrt(d), k, 0, [], 0

    elif lamb >= 1:
        # hardcode all deterministic case

        #Need to find top k values

        obj_bias = np.sum(np.array([d[d_arg_sort[i]] for i in range(n-k)]) ** 2)
        obj_var = 0

        arg_min_ordered, est_N_min, cutoff_min, determ_resorted, obj_bias_min, obj_var_min = np.zeros(n), 0, n, d_arg_sort[-k:], obj_bias, obj_var

    else:
        # First we iterate on j from 0 to k
        # For each one we want to calculate the break off point p such that a_i = 0 for i < p


        dSquare = d_sorted * d_sorted

        # sum of deterministic tail
        dTail = sum(dSquare[-k:])

        # Keep track of best j, p, objective
        # Start with j = 0 (all deterministic)
        est_N_min = 0
        cutoff_min = n
        arg_min = [0] * n
        obj_bias_min = sum(dSquare[:-k])
        obj_var_min = 0
        obj_min = obj_bias_min

        if lamb <= thresh:
            obj_min = float('inf') # Handles the case where we optimize instead over variance

        for est_N in range(1, k + 1):
            if est_N < k:
                arg, cutoff, obj_bias, obj_var = _mixed_objective_sorted_diagonal_solver(d_sorted[:-k + est_N], lamb / est_N)
            else:
                arg, cutoff, obj_bias, obj_var = _mixed_objective_sorted_diagonal_solver(d_sorted, lamb / est_N)

            obj_var = (1.0 / est_N) * obj_var # Account for multiple random estimators

            # Use variance as objective to handle lambda = 0 case
            if lamb <= thresh:
                obj = obj_var
            else:
                obj = obj_bias + lamb*obj_var

            # keep track of minimum stats
            if obj < obj_min:
                arg_min = arg
                est_N_min = est_N
                cutoff_min = cutoff
                obj_min = obj
                obj_bias_min = obj_bias
                obj_var_min = obj_var

        # pad argument
        arg_min = np.pad(arg_min, (0, n - len(arg_min)), 'constant')

        # Unsort argument
        arg_min_ordered = np.array([arg_min[d_arg_sort_inv[i]] for i in range(len(arg_min))])

        #Determine indices of the deterministic entries
        determ = range(n - k + est_N_min, n)
        determ_resorted = np.array([d_arg_sort[i] for i in determ])



    return arg_min_ordered, est_N_min, cutoff_min, determ_resorted, obj_bias_min, obj_var_min





# Designed so one can do use different lambda and k projection set ups without recomputing the SVD of the covariance
# Inputs:   covariance_X -   the covariance matrix of X
#           covariance_W -   the covariance matrix of W
# Outputs:  scale_x -        Uxw^T . Qx^{-T}, the linear transformation applied to the x space
#           scale_w -        Vxw^T . Qw^{-T}, the linear transformation applied to the w space
#           Sxw -           The singular values of Qx . Qw^T sorted largest to smallest
# Notes: uses svd as blackbox commands
#        faster relaxation available in quickMixedEstimators which avoids these matrix calculations
def preprocess_optimal_mixed_estimators(covariance_X, covariance_W):
    n = covariance_W.shape[0]
    # check sizes of covariance matrices to be square and same
    if covariance_W.shape != (n, n) or covariance_X.shape != (n, n):
        raise ValueError('Dimension Incorrect, covariance either not square or not the same size')

    # Factor covariance into square roots, uses svd to avoid errors (Sqrtm not cooperating)
    # Here we are assuming that covarianceX, covarianceW are symmetric matrices
    (Ux, Sx, VxT) = linal.svd(covariance_X)
    Qx = np.transpose(VxT) @ np.diag(np.sqrt(Sx)) @ VxT
    (Uw, Sw, VwT) = linal.svd(covariance_W)
    Qw = np.transpose(VwT) @ np.diag(np.sqrt(Sw)) @ VwT
    (Uxw, Sxw, VxwT) = np.linalg.svd(Qx @ np.transpose(Qw))  # Sxw 1-D numpy array sorted, Vxw already transposed


    scale_x = np.transpose(Uxw) @ np.linalg.pinv(np.transpose(Qx))
    scale_w = VxwT @ np.linalg.pinv(np.transpose(Qw))

    return scale_x, scale_w, Sxw



# TODO: There is unnecessary sorting all over the place (the solver gets called with singular values which are sorted)
# Takes in covariance matrices of two distributions and other parameters to find the optimal mix of random and deterministic
# estimators to minimized mixed objective
# Inputs:   preprocess_SVD -   (xScale, wScale, Sxw) from preprocessOptimalMixedEstimators
#           k -                 The target dimension of the projection
#           lamb -              Parameter controlling our mixed objective
# Outputs: estimatorTupleX, estimatorTupleW
#           An estimatorTuple consists of the following:
#               rand_Ax -   n by n matrix so random estimators are (randAx x)^T R  (R random iid gaussian)
#               determ_Ax - k-j by n matrix so our determistic estimators for our x space are (determAx x)^T
#                           Note: determ_Ax will be empty matrix if all random estimators, use vstack to concat
#               est_N -     integer between 0 and k indicating number of random estimators
#           cutoff -    the number of zeroed indices in randAx, randAw
#           obj_bias -  the minimal bias
#           obj_var -   the minimal variance
# Notes: Uses np, scipy matrix sqrt and svd, and then called the diagonal solver.
def optimal_mixed_estimators(preprocess_SVD, k, lamb):
    scale_x, scale_w, Sxw = preprocess_SVD
    n = scale_x.shape[0]
    Sxw_sqrt = np.sqrt(Sxw)

    # TODO: Sxw is already sorted largest to smallest, if could reverse and update everything then could send straight
    # TODO: to the internal command
    arg_min, est_N, cutoff, determ_coords, obj_bias, obj_var = mixed_estimator_diagonal_solver(Sxw, lamb, k)
    A = np.diag(arg_min)

    determ_Ax = np.diag(Sxw_sqrt) @ scale_x
    determ_Aw = np.diag(Sxw_sqrt) @ scale_w

    if len(determ_coords) > 0:
        determ_Ax = determ_Ax[determ_coords, :]  # select correct singular components for determinstic part
        determ_Aw = determ_Aw[determ_coords, :]  # select correct singular components for determinstic part
    else:
        # Empty array with n columns, need the columns to concatenate arrays later
        determ_Ax = np.array([], dtype=np.int64).reshape(0, n)
        determ_Aw = np.array([], dtype=np.int64).reshape(0, n)

    rand_Ax = A @ scale_x
    rand_Aw = A @ scale_w

    return (rand_Ax, determ_Ax, est_N), (rand_Aw, determ_Aw, est_N), cutoff, obj_bias, obj_var



# Takes in covariance matrices of two distributions and other parameters to find the naive mix of random and deterministic
# estimators to minimized mixed objective. Here naive means top principal components are deterministic, rest are random
# Inputs:   preprocess_SVD -   (xScale, wScale, Sxw) from preprocessOptimalMixedEstimators
#           k -                 The target dimension of the projection
#           lamb -              Parameter controlling our mixed objective
# Outputs: [estimatorTupleX, estimatorTupleW]_i
#           An estimatorTuple consists of the following:
#               rand_Ax -   n by n matrix so random estimators are (randAx x)^T R  (R random iid gaussian)
#               determ_Ax - k-j by n matrix so our determistic estimators for our x space are (determAx x)^T
#                           Note: determ_Ax will be empty matrix if all random estimators, use vstack to concat
#               est_N -     integer between 0 and k indicating number of random estimators
# Notes: Uses np, scipy matrix sqrt and svd, and then called the diagonal solver.
def naive_mixed_estimators(preprocess_SVD, k):
    scale_x, scale_w, Sxw = preprocess_SVD
    print(scale_x)
    n = scale_x.shape[0]
    Sxw_sqrt = np.sqrt(Sxw)

    naive_estimators = [0] * (k+1)

    for est_N in range(0, k + 1):

        arg_min_head = [0] * (k - est_N)
        arg_min_tail = Sxw_sqrt[k - est_N:].tolist()
        arg_min = np.array(arg_min_head + arg_min_tail)

        determ_coords = range(k - est_N)


        A = np.diag(arg_min)

        determ_Ax = np.diag(Sxw_sqrt) @ scale_x
        determ_Aw = np.diag(Sxw_sqrt) @ scale_w

        if len(determ_coords) > 0:
            determ_Ax = determ_Ax[determ_coords, :]  # select correct singular components for determinstic part
            determ_Aw = determ_Aw[determ_coords, :]  # select correct singular components for determinstic part
        else:
            # Empty array with n columns, need the columns to concatenate arrays later
            determ_Ax = np.array([], dtype=np.int64).reshape(0, n)
            determ_Aw = np.array([], dtype=np.int64).reshape(0, n)

        rand_Ax = A @ scale_x
        rand_Aw = A @ scale_w
        naive_estimators[est_N] = (rand_Ax, determ_Ax, est_N), (rand_Aw, determ_Aw, est_N)

    return naive_estimators




#TODO: update doc
# Designed so one can do use different lambda and k projection set ups without recomputing the SVD of the covariance
# Inputs:   covariance_X -   the covariance matrix of X
#           covariance_W -   the covariance matrix of W
# Outputs:  scale_x -        Uxw^T . Qx^{-T}, the linear transformation applied to the x space
#           scale_w -        Vxw^T . Qw^{-T}, the linear transformation applied to the w space
#           Sxw -           The singular values of Qx . Qw^T sorted largest to smallest
# Notes: uses svd as blackbox commands
#        faster relaxation available in quickMixedEstimators which avoids these matrix calculations
def preprocess_quick_mixed_estimators(covariance_diag_x, covariance_diag_w):

    n = covariance_diag_x.shape[0]

    # check sizes of covariance matrices to be square and same
    if covariance_diag_w.shape != (n, ) or covariance_diag_x.shape != (n, ):
        raise ValueError('Dimension Incorrect, covariance diagonal not one dimensional or not the same size')

    covariance_diag_x_sqrt = np.sqrt(covariance_diag_x)
    covariance_diag_w_sqrt = np.sqrt(covariance_diag_w)

    scale_x = np.linalg.pinv(np.diag(covariance_diag_x_sqrt))
    scale_w = np.linalg.pinv(np.diag(covariance_diag_w_sqrt))

    return scale_x, scale_w, covariance_diag_w_sqrt * covariance_diag_x_sqrt




#TODO: should output estimator tuple, in process of removing this, moving quick to preprocess step
# Takes in covariance matrix diagonals of two distributions and other parameters to find the optimal mix of random
# and deterministic estimators to minimized mixed objective. Only allows diagonal transformations.
# Uses the fact the determnistic part will be a diagonal projection to compress output
# Inputs:   covarianceDiagX -   The diagonal of the covariance matrix of our first distribution
#           covarianceDiagW -   The diagonal of the covariance matrix of our second distribution
#           k -                 The target dimension of the projection
#           lamb -              Parameter controlling our mixed objective
# Outputs:
#           rand_Ax -       n by 1 matrix so random estimators are (diag(randAx) x)^T R  (R random iid gaussian)
#           rand_Aw -       n by 1 matrix so random estimators are (diag(randAw) w)^T R  (R random iid gaussian)
#           determ_ind -    list of indices (0 to n) such that deterministic for both is (diag(1_determAx) x)^T
#           est_N -         integer between 0 and k indicating number of random estimators
#           cutoff -        integer representing the number of zeroed coordinates in rand_Ax or rand_Aw
# Notes: Uses np sqrt and calls the diagonal solver
#        Doesn't spit out estimator tuple since required space for tuple is O(n^2)
#        Doesn't output obj_bias, obj_var since they are the objectives for the relaxed problem
#TODO: need to add pinv for singular covariance
# def quick_mixed_estimators(covariance_diag_x, covariance_diag_w, k, lamb):
#     n = covariance_diag_x.shape[0]
#
#     # check sizes of covariance matrices to be square and same
#     if covariance_diag_w.shape != (n, ) or covariance_diag_x.shape != (n, ):
#         raise ValueError('Dimension Incorrect, covariance diagonal not one dimensional or not the same size')
#
#     covariance_diag_x_sqrt = np.sqrt(covariance_diag_x)
#     covariance_diag_w_sqrt = np.sqrt(covariance_diag_w)
#
#     preprocess_SVD = (np.diag(np.power(covariance_diag_x_sqrt, -1)), np.diag(np.power(covariance_diag_w_sqrt,-1)),np.diag(covariance_diag_w_sqrt * covariance_diag_x_sqrt))
#
#     #
#     # # Factor covariance into square roots:
#     # Qx = np.sqrt(covariance_diag_x)
#     # Qw = np.sqrt(covariance_diag_w)
#     #
#     # d = Qx * Qw
#     #
#     # arg_min, est_N, cutoff, determ_indices, obj_bias, obj_var = mixed_estimator_diagonal_solver(d, lamb, k)
#     # rand_Ax = arg_min / Qx
#     # rand_Aw = arg_min / Qw
#
#     return optimal_mixed_estimators(preprocess_SVD, k, lamb)


#Given covariance matrices, evaluated E Bias^2 + \lambda E Variance

def _all_random_estimator_tuple(n,k):
    return np.identity(n), np.array([], dtype=np.int64).reshape(0, n), k

def _evaluateObjective(covarianceX, covarianceW, estimator_tuple_X, estimator_tuple_W, k, lamb):
    Qx = linal.sqrtm(covarianceX)
    Qw = linal.sqrtm(covarianceW)

    n = covarianceX.shape[0]

    randomComponent = np.zeros((n, n))
    determComponent = np.zeros((n, n))
    variance = 0

    #Unpack estimator tuple
    randAx, determAx, randEstCount = estimator_tuple_X
    randAw, determAw, randEstCount = estimator_tuple_W

    determEstCount = k - randEstCount


    #Handles case where there are random estimators
    if randEstCount > 0:
        randomXComponent = Qx @ np.transpose(randAx)
        randomWComponent = randAw @ np.transpose(Qw)
        randomComponent = randomXComponent @ randomWComponent
        variance = (lamb/randEstCount) * (linal.norm(randomComponent)**2 + (linal.norm(randomWComponent) * linal.norm(randomXComponent)) **2)

    #Handles case where there are deterministic estimators
    if determEstCount > 0:
        determComponent = Qx @ np.transpose(determAx) @ determAw @ np.transpose(Qw)

    bias = linal.norm(randomComponent + determComponent - Qx @ np.transpose(Qw)) ** 2

    return bias + variance




## Testers

# def mixed_objective_diagonal_solver(d, lamb):
# def mixed_estimator_diagonal_solver(d, lamb, k, thresh=0):
# def preprocess_optimal_mixed_estimators(covariance_X, covariance_W):
# def optimal_mixed_estimators(preprocess_SVD, k, lamb):
# def quick_mixed_estimators(covariance_diag_x, covariance_diag_w, k, lamb):



# def mixed_objective_diagonal_solver(d, lamb)
# return arg_min, cutoff_min, obj_bias, obj_var

def test_mixed_objective_diagonal_solver():
    import NumericalProjectionSolver as nps
    print("Prints differences of draws to the orp solver and a numerical solver of both arguements and objectives")
    print("Should be lists of differences near epsilon")
    n = 10  # ambient dimension
    N = 4  # Increment for lambda test
    d = npr.rand(n)
    for i in range(0, N):
        lamb = i / N
        arg_min, cutoff_min, obj_bias, obj_var = mixed_objective_diagonal_solver(d, lamb)
        return_num_tuple = nps.objective_minimizer(d, lamb)
        arg_min_num = return_num_tuple['x']
        obj_min_num = return_num_tuple['fun']
        print("arg_min differences: %s" % str(arg_min - arg_min_num))
        print("obj differences: %s" % str(obj_min_num - (obj_bias + lamb * obj_var)))


# def mixed_estimator_diagonal_solver(d, lamb, k, thresh=0):
# return arg_min_ordered, est_N_min, cutoff_min, determ_resorted, obj_bias_min, obj_var_min
def test_mixed_estimator_diagonal_solver():
    import NumericalProjectionSolver as nps
    n = 10
    N = 4
    k = 3
    d = np.sort(npr.rand(n))
    #WARNING: d must be sorted
    for i in range(0, N):
        lamb = i / N +.0001
        arg_min, est_N, cutoff, determ, obj_bias, obj_var = mixed_estimator_diagonal_solver(d, lamb, k)
        return_num_tuple = nps.objective_minimizer_mixed(d, lamb, k)
        arg_min_num = return_num_tuple['x']
        obj_min_num = return_num_tuple['fun']
        print("arg_min differences: %s" % str(arg_min - arg_min_num))



# def preprocess_optimal_mixed_estimators(covariance_X, covariance_W):
# return scale_x, scale_w, Sxw
def test_preprocess_optimal_mixed_estimators():
    # Toy Example: Results on diagonal symmetric case
    # covarianceX = covarianceW = D is a diagonal matrix
    # preprocess_optimal_mixed_estimators(D,D) = (D^{-1/2}, D^{-1/2}, D)
    n = 10
    d_x = np.flip(np.sort(npr.rand(n)),0)
    d_w = np.flip(np.sort(npr.rand(n)),0)
    D_x = np.diag(d_x)
    D_w = np.diag(d_w)
    test_results = preprocess_optimal_mixed_estimators(D_x, D_w)
    expected_results = (np.diag(np.power(d_x, -.5)), np.diag(np.power(d_w, -.5)), np.power(d_x, .5) * np.power(d_w, .5))
    for i in range(3):
        print(np.amax(test_results[i] - expected_results[i]))


# def optimal_mixed_estimators(preprocess_SVD, k, lamb):
# return (rand_Ax, determ_Ax, est_N), (rand_Aw, determ_Aw, est_N), cutoff, obj_bias, obj_var
#def test_optimal_mixed_estimators():
#    return None


# def quick_mixed_estimators(covariance_diag_x, covariance_diag_w, k, lamb):

#
# #TODO: outdated
# def testmixedEstimatorDiagonalRandom(n, m, lamb = npr.rand()):
#     d = .1 * npr.rand(n)
#     d = np.sort(d)
#
#     print("lambda: %s" % lamb)
#     executeTime = time.process_time()
#     calculatedMinEst = mixed_estimator_diagonal_solver(d, lamb, m)
#     print("Execute Time: %s" % (time.process_time() - executeTime))
#     print("minimizing arg: %s" % calculatedMinEst[0])
#     print("Number of Random Estimators: %s" % calculatedMinEst[1])
#     print("Number of zeroed coordinates: %s" % calculatedMinEst[2])
#     print("Indices of determistic coords: %s" % calculatedMinEst[3])
#     print("Minimal value of Obj: %s" % calculatedMinEst[4])
#     print("Value of Obj for all deterministic: %s" % sum((d * d)[:-m]))
#     j = calculatedMinEst[1]
#     if j > 0:
#         if m - j == 0:
#             newd = d
#         else:
#             newd = d[:-(m - j)]
#         print("Numerical minimization with same division of estimators: %s" %
#               fractionalObjectiveMinimizer(newd, lamb / j)["fun"])
#
#
# #n - dimension
# #k - target dimension
# #N - number of trials
# #all diagonal tests
# def testMixedEstimatorFunctions(n, k, N=10):
#     n = 100
#     k = 10
#     for i in range(N):
#         d1 = npr.rand(n)
#         d2 = npr.rand(n)
#         lamb = npr.rand()
#         # randomMatrix = npr.rand(n, n)
#         # randomPSD = randomMatrix @ np.transpose(randomMatrix)
#         covarianceX = np.diag(d1)
#         covarianceW = np.diag(d2)
#         # covarianceX = randomPSD
#         # covarianceW = randomPSD
#
#         diagonalSolverSoln = mixed_estimator_diagonal_solver(np.sqrt(d1 * d2), lamb, k)
#         sol = optimalMixedEstimators(preprocessOptimalMixedEstimators(covarianceX, covarianceW), k, lamb)
#         diagSol = quickMixedEstimators(np.diag(covarianceX), np.diag(covarianceW), k, lamb)
#         optimalEstimatorEval = _evaluateObjective(covarianceX, covarianceW, sol[0], sol[1],
#                                                  sol[2], sol[3], lamb, sol[4], k - sol[4])
#
#         diagProj = np.zeros(n)
#         for index in diagSol[2]:
#             diagProj[index] = 1
#         diagProj = np.diag(diagProj)
#         optimalDiagEstimatorEval = _evaluateObjective(covarianceX, covarianceW, np.diag(diagSol[0]), np.diag(diagSol[1]),
#                                                      diagProj, diagProj, lamb, diagSol[3], k - diagSol[3])
#
#
#
#
#         print("direct solution from diagonal solver: \t\t\t\t%s" % diagonalSolverSoln[-1])
#         print("solution from full solver optimalMixedEstimators: \t%s" % optimalEstimatorEval)
#         print("solution from quick solver quickMixedEstimators: \t%s" % optimalDiagEstimatorEval)
#         print("-")
#
#
