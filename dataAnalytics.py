import numpy as np
import pickle
import plotter
import ProjectionComputer as pc
import OptimalRandomProjections as orp
import helper as h
import regressors


#All outputs are 1-D numpy arrays with [mean, std]. If no std known, returns 0 as std

#The original data is formated so the inputs dataX, dataW satisfy dataX[i], dataW[i] is the ith data point
#The proj data is formated so dataProjX[i,j] is the ith data point under the jth realization of our projection



#Distribution Analytics:

#Takes in 2D array where data[i] is ith data point and makes it mean zero
def center_data(data):
    data = np.array(data)
    mean = np.mean(data, 0)
    return data - mean[None, :]

#Notes: just applies np.cov
#Input: N x n matrix where each row is a data sample
def emperical_covariance(data):
    data = center_data(data)
    empericalCov = np.cov(np.transpose(data))
    return empericalCov

#Input: N x n matrix where each row is a data sample
def emperical_covariance_diag(data):
    data = center_data(data)
    return np.array([data[:, i].dot(data[:, i]) for i in range(data.shape[1])]) / (data.shape[0] - 1)



#Preprocessor:

#returns array where data_proj_diff[i] is a 2d array where (i,j)the entry is <proj(x_i), proj(w_j)> - <x_i, w_j>
def compute_distortions(dataX, dataW, dataXProj, dataWProj):
    #get relevant dimensions
    (sizeData, sizeRealizations, sizeSpace) = dataXProj.shape

    data_dot = dataX @ np.transpose(dataW) # (i,j)th entry is <dataX[i], dataW[j]>

    data_proj_diff = np.empty((sizeRealizations, sizeData, sizeData)) # data_proj_dot[i] is the dot product array for the ith realization
    for i in range(sizeRealizations):
        data_proj_diff[i] = np.abs(dataXProj[:, i, :] @ np.transpose(dataWProj[:, i, :]) - data_dot)

    return data_proj_diff


#Data Analytics:


# Computes E_{R} max_{x,w} |<proj_R(x), proj_R(w)> - <x,w>|
def max_distortion(processed_distortions):
    max_distortions = np.amax(np.abs(processed_distortions), axis=(1, 2))
    return np.array([np.mean(max_distortions), np.std(max_distortions)])


def l2_distortion(processed_distortions):
    avg_distortions = np.mean(np.power(np.abs(processed_distortions), 2), axis=(1, 2))
    return np.array([np.mean(avg_distortions), np.std(avg_distortions)])


def p99_distortion(processed_distortions):
    p99_distortions = np.percentile(np.abs(processed_distortions), 99, axis=(1, 2))
    return np.array([np.mean(p99_distortions), np.std(p99_distortions)])

def emperical_bias(processed_distortions):
    biases = np.power(np.mean(processed_distortions, axis=0), 2)

    return np.array([np.mean(biases), np.std(biases)])



### BATCH COMMANDS
# FMM
def generate_fmm_analytics(data_X, data_W, realizations_N = 100):
    n = data_X.shape[1]
    samples_N = data_X.shape[0]

    #center data
    data_X = center_data(data_X)
    data_W = center_data(data_W)

    #compute emperical covariance
    emp_covariance_X = emperical_covariance(data_X)
    emp_covariance_W = emperical_covariance(data_W)

    preprocess = orp.preprocess_optimal_mixed_estimators(emp_covariance_X, emp_covariance_W)
    preprocess_quick = orp.preprocess_quick_mixed_estimators(np.diag(emp_covariance_X), np.diag(emp_covariance_W))


    data_analytics = np.empty((0,4,2))

    #TODO: REMOVE HARDCODED RANGE
    for k in range(5, 25):
        print(k)

        #compute estimators
        estimator_X, estimator_W, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(preprocess, k, -1)
        estimator_quick_X, estimator_quick_W, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(preprocess_quick, k, -1)
        estimator_random = orp._all_random_estimator_tuple(n, k)


        #realize optimal estimators
        realizations_X, realizations_W = pc.create_random_projection_realizations(data_X, estimator_X, data_W, estimator_W, realizations_N)
        # realizations_W = pc.createRandomProjectionRealizations(data_W, estimator_W, realizationsN)
        pc.verify_realizations(data_X, data_W, realizations_X, realizations_W, estimator_X, estimator_W)

        #realize quick estimators
        realizations_quick_X, realizations_quick_W = pc.create_random_projection_realizations(data_X, estimator_quick_X, data_W, estimator_quick_W, realizations_N)
        # realizations_quick_W = pc.createRandomProjectionRealizations(data_W, estimator_quick_W, realizationsN)
        pc.verify_realizations(data_X, data_W, realizations_quick_X, realizations_quick_W, estimator_quick_X, estimator_quick_W)

        #realize purely random estimators
        realizations_rand_X, realizations_rand_W = pc.create_random_projection_realizations(data_X, estimator_random, data_W, estimator_random, realizations_N)
        # realizations_rand_W = pc.createRandomProjectionRealizations(data_W, estimator_random, realizationsN)
        pc.verify_realizations(data_X, data_W, realizations_rand_X, realizations_rand_W, estimator_random, estimator_random)

        #compute distortions
        distortions = compute_distortions(data_X, data_W, realizations_X, realizations_W)
        distortions_quick = compute_distortions(data_X, data_W, realizations_quick_X, realizations_quick_W)
        distortions_rand = compute_distortions(data_X, data_W,  realizations_rand_X, realizations_rand_W)

        #compute frobenius norms
        l2_distortions = l2_distortion(distortions)
        l2_quick_distortions = l2_distortion(distortions_quick)
        l2_rand_distortions= l2_distortion(distortions_rand)

        #tic
        tic = np.array([k,0])

        #append data
        data_row = np.array([tic, l2_distortions, l2_quick_distortions, l2_rand_distortions])
        data_analytics = np.vstack((data_analytics, [data_row]))
    return data_analytics


def fmm_filestring(filename, n, samples_N, realizations_N):
    file_string_components = [filename, str(n), str(samples_N), str(realizations_N)]
    file_string = "_".join(file_string_components)

    return file_string

def save_fmm_analytics(data_analytics, file_string):

    #save data analytics and call plotter
    pickle.dump(data_analytics, open("data_analytics_fmm/" + file_string + ".p", "wb+"))
    plotter.generate_fmm_plots(file_string)
    print(file_string)

def generate_save_plot_fmm_analytics(data_X, data_W, filename, realizations_N = 100):
    #extract dimensions
    n = data_X.shape[1]
    samples_N = data_X.shape[0]

    data_analytics = generate_fmm_analytics(data_X, data_W, realizations_N)

    filestring = fmm_filestring(filename, n, samples_N, realizations_N)

    save_fmm_analytics(data_analytics, filestring)

    plotter.generate_fmm_plots(filestring)

# Interpolation


'''
Generates different data analytics for symmetric data with lambda from zero to one
Inputs:
data -              samples_N x ambient_dim sized array with data samples      
target_dim -        target dimension to project to, should be integer between 1 and ambient_dim
realizations_N -    number of random projections to realize for analytics
Delta -             how many divisions for lambda between 0 and 1
bias_var -          a mostly useless boolean which decides whether to measure emperical bias/variance
Outputs:
data_analytics -    m x 10 x 2 array 
                    first axis:     correponds to each different lambda sampled
                    second axis:    corresponds to each different analytic sampled in the order:
                                    tic_sample, emp_bias_sample, emp_var_sample, bias_sample, var_sample,
                                    max_distortion, p99_distortion, l2_distortion, estimator_N, cutoff
                    third axis:     mean, std, if std doesnt make sense then its zero
'''
def generate_interpolation_analytics(data, target_dim, realizations_N, Delta, bias_var=False):

    data = center_data(data) # normalize mean to zero

    emp_cov = emperical_covariance(data)

    preprocess = orp.preprocess_optimal_mixed_estimators(emp_cov, emp_cov)

    naive_estimators = orp.naive_mixed_estimators(preprocess, target_dim)

    naive_max_distortion =[0.0] *  len(naive_estimators)
    naive_l2_distortion = [0.0] *  len(naive_estimators)
    for i in range(len(naive_estimators)):
        estimator = naive_estimators[i]
        naive_estimator_realizations = pc.create_random_projection_realizations(data, estimator[0], data, estimator[1], realizations_N)
        distortions = compute_distortions(data,data,naive_estimator_realizations[0], naive_estimator_realizations[1])
        naive_max_distortion[i] = max_distortion(distortions)
        naive_l2_distortion[i] = l2_distortion(distortions)

    min_max_distortion = np.min(np.array(naive_max_distortion)[1:, 0])
    min_l2_distortion = np.min(np.array(naive_l2_distortion)[1:, 0])




    data_analytics = np.empty((0, 12, 2))

    end_tick = Delta
    tic = 0
    while tic < end_tick:
        lamb = (tic)*1.0/(Delta -1)
        print(lamb)

        #Get the estimator tuples for optimal projections
        estimatorTupleX, estimatorTupleW, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(preprocess, target_dim, lamb)

        #Realize optimal projections
        projSamples_X, projSamples_W = pc.create_random_projection_realizations(data, estimatorTupleX, data, estimatorTupleW, realizations_N)

        #Compute distortions for analytics
        distortions = compute_distortions(data, data, projSamples_X, projSamples_W)

        # compute data analytics
        tic_sample = np.array([tic, 0])

        emp_bias_sample = np.array([0, 0])
        emp_var_sample = np.array([0, 0])
        if bias_var:
            emp_bias_sample = emperical_bias(distortions)
            emp_var_sample = empericalVariance(projSamples_X, projSamples_X)


        bias_sample = np.array([obj_bias, 0])
        var_sample = np.array([obj_var, 0])

        max_distortion_sample = max_distortion(distortions)
        p99_distortion_sample = p99_distortion(distortions)
        l2_distortion_sample = l2_distortion(distortions)
        est_N = estimatorTupleX[-1]
        estimator_N = np.array([est_N, 0]) #naive_max_distortion[est_N]
        min_max_distortion_N = np.array([min_max_distortion,0])
        min_l2_distortion_N = np.array([min_l2_distortion,0])
        cutoff = np.array([cutoff,0])

        # store data analytics as row in array
        data_row = np.array(      [tic_sample,
                                   emp_bias_sample, emp_var_sample,
                                   bias_sample, var_sample,
                                   max_distortion_sample, p99_distortion_sample, l2_distortion_sample,
                                   estimator_N, cutoff, min_max_distortion_N, min_l2_distortion_N])

        data_analytics = np.vstack((data_analytics, [data_row]))

        #If determinstic, tic 3 more times
        if est_N == 0 and end_tick - tic > 3:
            end_tick = tic + 3

        tic += 1


    # pickle.dump((tics, np.array(empBiasSamples), empVarSamples, np.array(biasSamples), np.array(varianceSamples), maxDistortion, avgDistortion, list_estimator_N), open("data_analytics/" + filename + ".p", "wb+"))
    return data_analytics


def interpolation_filestring(filename, k, samples_N, realizations_N):
    file_string_components = [filename, str(k), str(samples_N), str(realizations_N)]
    file_string = "_".join(file_string_components)
    return file_string

def save_interpolation_analytics(data_analytics, file_string):
    pickle.dump(data_analytics, open("data_analytics/" + file_string + ".p", "wb+"))
    print(file_string)


def generate_save_plot_interpolation_analytics(drawSamples, k, filename, realizations_N = 100, Delta = 100, bias_var=False):
    data_analytics = generate_interpolation_analytics(drawSamples, k, realizations_N, Delta, bias_var)

    samples_N = drawSamples.shape[0]

    file_string = interpolation_filestring(filename, k, samples_N, realizations_N)

    save_interpolation_analytics(data_analytics, file_string)

    plotter.generate_interpolation_plots(file_string, bias_var)


# Regression

#TODO: FINISH SECTION

def generate_regression_analytics(data_regress, data_values, realizations_N=100, test_frac = .1):
        ambient_dim = data_regress.shape[1]
        samples_N = data_regress.shape[0]

        # split into test and train
        test_samples_N = np.int(np.floor(samples_N * test_frac))
        test_data = data_regress[:test_samples_N, :]
        test_data_values = data_values[:test_samples_N]
        train_data = data_regress[test_samples_N:, :]
        train_data_values = data_values[test_samples_N:]

        # calculate covariance of train
        train_data = center_data(train_data)
        cov = emperical_covariance(train_data)
        cov_diag = np.diag(cov)


        data_analytics = np.empty((0, 10, 2))

        for k in range(5,25):
            print("k %s" % str(k))
            data_row = np.empty((10,2))

            for lamb_ind in range(9):
                lamb = lamb_ind*.25 - 1
                print("lamb %s" % str(lamb))

                #Calculate estimator with Var^lamb
                est = (np.diag(h.psuedo_power(cov_diag, lamb)), np.array([], dtype=np.int64).reshape(0, ambient_dim), k)

                # project all of the data
                realizations_X, realizations_W = pc.create_random_projection_realizations(data_regress, est,
                                                                                          data_regress, est,
                                                                                          realizations_N)
                train_realizations = realizations_X[test_samples_N:, :, :]
                test_realizations = realizations_X[:test_samples_N, :, :]

                data_row[lamb_ind+1] = regressors.regress_realizations_error(train_realizations, train_data_values, test_realizations,
                                                    test_data_values)
            # tic
            tic = np.array([k, 0])
            data_row[0] = tic

            data_analytics = np.vstack((data_analytics, [data_row]))
        return data_analytics


def regression_filestring(filename, n, samples_N, realizations_N):
    file_string_components = [filename, str(n), str(samples_N), str(realizations_N)]
    file_string = "_".join(file_string_components)
    return file_string

def save_regression_analytics(data_analytics, file_string):
    pickle.dump(data_analytics, open("data_analytics_regress/" + file_string + ".p", "wb+"))
    print(file_string)


def generate_save_plot_regression_analytics(data_regress, data_values, filename, realizations_N = 100, test_frac=.1):
    data_analytics = generate_regression_analytics(data_regress, data_values, realizations_N, test_frac)

    samples_N = data_regress.shape[0]
    ambient_dim = data_regress.shape[1]

    file_string = regression_filestring(filename, ambient_dim, samples_N, realizations_N)

    save_regression_analytics(data_analytics, file_string)

    plotter.generate_regression_plots(file_string)

#lambda interp


def generate_regression_analytics_lamb(data_regress, data_values, realizations_N=100, test_frac = .1, target_dim=5):
        ambient_dim = data_regress.shape[1]
        samples_N = data_regress.shape[0]
        # split into test and train
        test_samples_N = np.int(np.floor(samples_N * test_frac))

        test_data = data_regress[:test_samples_N, :]
        test_data_values = data_values[:test_samples_N]
        train_data = data_regress[test_samples_N:, :]
        train_data_values = data_values[test_samples_N:]

        print(train_data.shape)
        # calculate covariance of train
        cov_diag = emperical_covariance_diag(train_data)
        print(np.sort(cov_diag))

        data_analytics = np.empty((0, 6, 2))

        for lamb_ind in range(0, 13):
            lamb = lamb_ind*.25 - 1.5
            data_row = np.empty((6,2))
            # tic
            tic = np.array([lamb, 0])
            data_row[0] = tic
            print("lamb %s" % str(lamb))

            for sub_ind in range(1,6):
                print("k %s" % str(5*sub_ind))
                target_dim = 5*sub_ind


                #Calculate estimator with Var^lamb
                est = (np.diag(h.psuedo_power(cov_diag, lamb)), np.array([], dtype=np.int64).reshape(0, ambient_dim), target_dim)
                print(np.sort(h.psuedo_power(cov_diag, lamb)))

                print("projecting...")
                # project all of the data
                realizations_X, realizations_W = pc.create_random_projection_realizations(data_regress, est,
                                                                                          data_regress, est,
                                                                                          realizations_N, is_diagonal=True)
                train_realizations = realizations_X[test_samples_N:, :, :]
                test_realizations = realizations_X[:test_samples_N, :, :]
                print("regressing...")
                data_row[sub_ind] = regressors.regress_realizations_error(train_realizations, train_data_values, test_realizations,
                                                                     test_data_values)

            data_analytics = np.vstack((data_analytics, [data_row]))
        return data_analytics

def generate_save_plot_regression_analytics_lamb(data_regress, data_values, filename, realizations_N = 100, test_frac=.1, target_dim=5):
    data_analytics = generate_regression_analytics_lamb(data_regress, data_values, realizations_N, test_frac, target_dim=target_dim)

    samples_N = data_regress.shape[0]
    ambient_dim = data_regress.shape[1]

    file_string = regression_filestring(filename, ambient_dim, samples_N, realizations_N) + "_lamb_" +str(target_dim)

    save_regression_analytics(data_analytics, file_string)

    plotter.generate_regression_lamb_plots(file_string)



'''shows among all naive split of rand and dest max and l2 distortion'''
def generate_naive_plots(data, file_name, target_dim, realizations_N=20):
    data = center_data(data) # normalize mean to zero

    emp_cov = emperical_covariance(data)

    preprocess = orp.preprocess_optimal_mixed_estimators(emp_cov, emp_cov)

    naive_estimators = orp.naive_mixed_estimators(preprocess, target_dim)

    naive_max_distortion =[0.0] * len(naive_estimators)
    naive_l2_distortion = [0.0] * len(naive_estimators)
    for i in range(len(naive_estimators)):
        estimator = naive_estimators[i]
        naive_estimator_realizations = pc.create_random_projection_realizations(data, estimator[0], data, estimator[1], realizations_N)
        distortions = compute_distortions(data,data,naive_estimator_realizations[0],naive_estimator_realizations[1])
        naive_max_distortion[i] = max_distortion(distortions)[0]
        naive_l2_distortion[i] = l2_distortion(distortions)[0]

    import matplotlib.pyplot as plt

    plt.title(file_name)
    plt.xlabel('est_N')
    plt.ylabel('normalized units')

    max_normalizer = max(naive_max_distortion)
    l2_normalizer = max(naive_l2_distortion)
    plt.plot(range(len(naive_estimators)), naive_max_distortion/max_normalizer, 'aqua', label='max')
    plt.plot(range(len(naive_estimators)), naive_l2_distortion/l2_normalizer, 'chartreuse', label='l2')


    plt.legend()
    plt.savefig("data_analytics/" + file_name + "_naive")

    # Plot max/avg distortion
    plt.gcf().clear()



'''shows the linear interpolation of naive l2 minimizer to PCA'''
def generate_naive_interp_plots(data, file_name, target_dim, realizations_N=20):
    data = center_data(data) # normalize mean to zero

    samples_N = data.shape[0]
    ambient_dim = data.shape[1]

    emp_cov = emperical_covariance(data)

    preprocess = orp.preprocess_optimal_mixed_estimators(emp_cov, emp_cov)

    naive_estimators = orp.naive_mixed_estimators(preprocess, target_dim)




    naive_max_distortion = np.array([0.0] * len(naive_estimators))
    naive_l2_distortion = np.array([0.0] * len(naive_estimators))
    distortions = np.empty((target_dim+1, realizations_N, samples_N, samples_N))
    for i in range(len(naive_estimators)):
        estimator = naive_estimators[i]
        naive_estimator_realizations_X, naive_estimator_realizations_W = pc.create_random_projection_realizations(data, estimator[0], data, estimator[1], realizations_N)
        distortions[i] = compute_distortions(data,data,naive_estimator_realizations_X, naive_estimator_realizations_X)
        naive_max_distortion[i] = max_distortion(distortions[i])[0]
        naive_l2_distortion[i] = l2_distortion(distortions[i])[0]


    naive_l2_min_arg = np.argmin(naive_l2_distortion[1:])

    naive_distortions = np.array(distortions[naive_l2_min_arg])
    pca_distortions = np.array(distortions[0])

    print(naive_l2_distortion[0])



    import matplotlib.pyplot as plt

    plt.title(file_name)
    plt.xlabel('lamb')
    plt.ylabel('normalized units')

    N = 100

    max_distortions = [0.0] * N
    l2_distortions = [0.0] * N

    for lamb_ind in range(N):
        lamb = lamb_ind*1.0/N
        print(lamb)
        max_distortions[lamb_ind] = max_distortion((1-lamb)*naive_distortions+lamb*pca_distortions)[0]
        l2_distortions[lamb_ind] = l2_distortion((1-lamb)*naive_distortions+lamb*pca_distortions)[0]

    max_normalizer = max(max_distortions)
    l2_normalizer = max(l2_distortions)
    plt.plot(range(N), max_distortions/max_normalizer, 'aqua', label='max')
    plt.plot(range(N), l2_distortions/l2_normalizer, 'chartreuse', label='l2')


    plt.legend()
    file_string = interpolation_filestring(file_name, target_dim,samples_N, realizations_N )
    plt.savefig("data_analytics/" + file_string + "_naive_interp")

    # Plot max/avg distortion
    plt.gcf().clear()








#TODO:Remove outdated code

# # Computes E_{R} max_{x,w} |<proj_R(x), proj_R(w)> - <x,w>|
# def computeMaxDistortion(dataX, dataW, dataXProj, dataWProj):
#     #get relevant dimensions
#     (sizeData, sizeRealizations, sizeSpace) = dataXProj.shape
#
#     expectedMaxDistortion = 0
#     for i in range(sizeRealizations):
#         maxDistortion = _computeMaxDistortionRealization(dataX, dataW, dataXProj[:, i, :], dataWProj[:, i, :])
#         expectedMaxDistortion = (1/(i+1))*((i)*expectedMaxDistortion + maxDistortion)
#
#     return expectedMaxDistortion
#
#
# #Computes max_{x,w} |<proj(x), proj(w)> - <x,w>|
# #here dataX, dataXProj are 2 dimensional, where both satisfy data[i] = ith data point
# #used for each projection realization slice
# def _computeMaxDistortionRealization(dataX, dataW, dataXProjRealization, dataWProjRealization):
#     #get relevant dimensions
#     (sizeData, sizeSpace) = dataXProjRealization.shape
#
#     maxDistortion = 0
#     for i in range(sizeData):
#         for j in range(sizeData):
#             point1 = dataX[i]
#             point2 = dataW[j]
#
#             point1Proj = dataXProjRealization[i]
#             point2Proj = dataWProjRealization[j]
#             dotProduct = point1.dot(point2)
#             dotProductProj = point1Proj.dot(point2Proj)
#             distortion = np.abs(dotProduct - dotProductProj)
#             if distortion > maxDistortion:
#                 maxDistortion = distortion
#     return maxDistortion
#
#
# # Computes E_{R}  |<proj_R(x), proj_R(w)> - <x,w>|_2^2
# def computeAverageDistortion(dataX, dataW, dataXProj, dataWProj):
#     #get relevant dimensions
#     (sizeData, sizeRealizations, sizeSpace) = dataXProj.shape
#
#     expectedAverageDistortion = 0
#     N = 0
#     for i in range(sizeRealizations):
#         averageDistortion = _computeAverageDistortionRealiation(dataX,dataW,dataXProj[:,i,:],dataWProj[:,i,:])
#         N += 1
#         expectedAverageDistortion = (1/N)*((N-1)*expectedAverageDistortion + averageDistortion)
#
#     return expectedAverageDistortion
#
#
#
# # Notes:    -could be adapted for a streaming model
# #           -using library commands like itertools and np.mean might be faster
# #           -dataXProjRealization two dimensional
# def _computeAverageDistortionRealiation(dataX, dataW, dataXProjRealization, dataWProjRealization):
#     #get relevant dimensions
#     (sizeData, sizeSpace) = dataXProjRealization.shape
#     averageDistortion = 0
#     N = 0
#     for i in range(sizeData):
#         for j in range(sizeData):
#             point1 = dataX[i]
#             point2 = dataW[j]
#
#             point1Proj = dataXProjRealization[i]
#             point2Proj = dataWProjRealization[j]
#             dotProduct = point1.dot(point2)
#             dotProductProj = point1Proj.dot(point2Proj)
#
#             distortion = (dotProduct - dotProductProj) ** 2
#
#             N += 1
#             #update average
#             averageDistortion = (1/N)*((N-1)*averageDistortion + distortion)
#     return averageDistortion
#
#
#
#
#
#
#
#
# # Computes E_{x,w} Bias^2 = E_{x,w} ( E_{R} <proj(x), proj(w)> - <x,w> )^2
# def empericalBiasSquared(dataX, dataW, dataXProj, dataWProj):
#
#     #get relevant dimensions
#     (sizeData, sizeRealizations, sizeSpace) = dataXProj.shape
#
#     empericalBiasSquared = 0
#     empericalBiasN = 0
#
#     for i in range(sizeData):
#         for j in range(sizeData):
#             averageDot = 0
#             averageN = 0
#             for k in range(sizeRealizations):
#                 dot = dataXProj[i, k].dot(dataWProj[j,k])
#                 averageN += 1
#                 averageDot = (1.0/averageN)*((averageN-1)*averageDot + dot)
#             bias = (averageDot - dataX[i].dot(dataW[j])) **2
#             empericalBiasN += 1
#             empericalBiasSquared = (1.0/empericalBiasN)*((empericalBiasN-1)*empericalBiasSquared + bias)
#
#     return empericalBiasSquared
#
#
# computes E_{x,w} ( E_R <proj_R(x), proj_R(w)>^2 - (E_R <proj_R(x), proj_R(w)>)^2 )
# first axis is over x, second axis over R


#TODO: Clean this up, use numpy methods instead
#TODO: Not high priority, variance/bias calcs not needed currently
def empericalVariance(dataXProj, dataWProj):

    #get relevant dimensions
    (sizeData, sizeRealizations, sizeSpace) = dataXProj.shape

    empericalVariance = 0
    empericalVarianceN = 0

    for i in range(sizeData):
        for j in range(sizeData):
            firstMoment = 0
            secondMoment = 0

            for k in range(sizeRealizations):
                dot = dataXProj[i, k].dot(dataWProj[j,k])

                firstMoment = (1/(k+1))*(k*firstMoment + dot)
                secondMoment = (1/(k+1))*(k*secondMoment + dot ** 2)


            variance = secondMoment - firstMoment ** 2
            empericalVarianceN += 1
            empericalVariance = (1/empericalVarianceN)*((empericalVarianceN-1)*empericalVariance + variance)

    return np.array([empericalVariance,0])


def theoreticalBiasSquared(preProcessedSVD, estimatorTupleX, estimatorTupleW):
    xScale, wScale, Sxw = preProcessedSVD
    randAx, determAx, randEstCount = estimatorTupleX
    randAw, determAw, randEstCount = estimatorTupleW

    xScale = np.linalg.pinv(np.transpose(xScale))
    wScale = np.linalg.pinv(np.transpose(wScale))

    n = xScale.shape[0]
    determCompon = np.zeros((n,n))
    if determAw.shape[0] > 0:
        determCompon = xScale @ np.transpose(determAx) @ determAw @ np.transpose(wScale)

    randCompon = xScale @ np.transpose(randAx) @ randAw @ np.transpose(wScale)

    return np.array([np.linalg.norm(randCompon+ determCompon - np.diag(Sxw)) **2,0])

def theoreticalVariance(preProcessedSVD, estimatorTupleX, estimatorTupleW):
    xScale, wScale, Sxw = preProcessedSVD
    randAx, determAx, randEstCount = estimatorTupleX
    randAw, determAw, randEstCount = estimatorTupleW
    xScale = np.linalg.pinv(np.transpose(xScale)) # makes xScale = Uxw^T Qx
    wScale = np.linalg.pinv(np.transpose(wScale)) # makes wScale = Vxw^T Qw

    result = 0
    if randEstCount > 0:
        result = (1/randEstCount)*(np.linalg.norm(xScale @ np.transpose(randAx) @ randAw @ np.transpose(wScale)) ** 2 +
                                   (np.linalg.norm(xScale @ np.transpose(randAx)) * np.linalg.norm(randAw @ np.transpose(wScale))) ** 2)

    return np.array([result,0])



