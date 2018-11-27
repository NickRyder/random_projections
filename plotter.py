import matplotlib.pyplot as plt
import pickle
import numpy as np
import helper as h

def generate_regression_plots(filename):
    data = pickle.load(open("data_analytics_regress/" + filename + ".p", "rb"))

    print(data.shape)
    # Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0,1)

    # unload data
    tics = data[0, :, :]
    lamb_means = data[1:, :, :]

    tics = tics[:, 0]

    plt.figure(1)

    plt.title(filename)
    plt.xlabel('target_dim')
    plt.ylabel('average frobenius error')

    # plt.plot(tics, list_estimator_N, 'ro',markersize=2,label='Random Estimator Count')
    lamb_N =lamb_means.shape[0]
    max_N = lamb_N
    for lamb_ind in range(max_N):
        color_interp = (lamb_ind*1.0)/(max_N-1)
        plt.plot(tics, lamb_means[lamb_ind, :, 0], color=[color_interp, 1-color_interp,0], label='Lambda: ' + str((2.0*lamb_ind)/(lamb_N-1) -1))
        plt.plot(tics, lamb_means[lamb_ind, :, 0] - lamb_means[lamb_ind, :, 1], '--', color=[color_interp, 1 - color_interp, 0])
        plt.plot(tics, lamb_means[lamb_ind, :, 0] + lamb_means[lamb_ind, :, 1], '--', color=[color_interp, 1 - color_interp, 0])

    plt.legend()
    plt.savefig("data_analytics_regress/" + filename + "_low")
    plt.gcf().clear()


def generate_regression_lamb_plots(filename):
    data = pickle.load(open("data_analytics_regress/" + filename + ".p", "rb"))

    print(data.shape)
    # Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0, 1)

    low_tic = 0
    data= data[:,low_tic:,:]

    # unload data
    tics = data[0, :, :]
    mse = data[1:, :, :]

    tics = tics[:, 0]
    print(tics)
    mse_N = mse.shape[0]

    plt.figure(1)

    plt.title(filename)
    plt.xlabel('lamb')
    plt.ylabel('sklearn score')
    for mse_ind in range(mse_N):
        color_interp = (mse_ind*1.0)/mse_N
        plt.plot(tics, mse[mse_ind, :, 0], color=[color_interp, 1 - color_interp, 0], label='k = ' + str(50*(mse_ind+ 1)))
        plt.plot(tics, mse[mse_ind, :, 0] - mse[mse_ind, :, 1], '--',
                 color=[color_interp, 1 - color_interp, 0])
        plt.plot(tics, mse[mse_ind, :, 0] + mse[mse_ind, :, 1], '--',
                 color=[color_interp, 1 - color_interp, 0])

    plt.legend()
    plt.savefig("data_analytics_regress/" + filename + "_" + str(low_tic))
    plt.gcf().clear()



def generate_interpolation_plots(filename, bias_var = False):
    data = pickle.load(open("data_analytics/" + filename + ".p", "rb"))

    print(data.shape)
    #Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0,1)


    #unload data
    [tics, emp_bias_samples, emp_var_samples, theor_bias_samples, theor_var_samples, max_distortion,
     p99_distortion, l2_distortion, list_estimator_N, cutoff, min_max_N, min_l2_N] = data

    print(emp_bias_samples.shape)


    bias_normalizer = max(theor_bias_samples[:, 0])
    theor_bias_samples = theor_bias_samples[:,0] / bias_normalizer

    var_normalizer = max(theor_var_samples[:, 0])
    theor_var_samples = theor_var_samples[:,0] / var_normalizer

    max_cutoff = max(cutoff[:,0])
    cutoff = cutoff[:,0] / max_cutoff

    normalizer = max(max_distortion[:, 0])
    max_distortion = max_distortion/normalizer
    p99_distortion = p99_distortion/normalizer


    list_estimator_N = list_estimator_N[:,0]/max(list_estimator_N[:,0])

    l2_normalizer = max(l2_distortion[:,0])
    l2_distortion = l2_distortion/l2_normalizer
    # max_distortion = _helper_normalize(max_distortion)
    # avg_distortion = _helper_normalize(avg_distortion)
    # p99_distortion = _helper_normalize(p99_distortion)

    min_max_N = min_max_N/normalizer
    min_l2_N = min_l2_N/l2_normalizer

    tics = tics[:, 0]

    if bias_var:
        emp_bias_means = h._helper_normalize(emp_bias_samples)[:, 0]

        emp_var = h._helper_normalize(emp_var_samples)[:, 0]

        theor_bias_samples = h._helper_normalize(theor_bias_samples)[:, 0]
        theor_var_samples = h._helper_normalize(theor_var_samples)[:, 0]
        plt.figure(1)

        #Plot bias/variance graph
        plt.title(filename)
        plt.xlabel('lambda')
        plt.ylabel('normalized units')

        plt.plot(tics, list_estimator_N, 'ro', markersize=2, label='Random Estimator Count')

        plt.plot(tics, theor_bias_samples, 'aqua', label='Theoretical Bias')
        plt.plot(tics, theor_var_samples, 'chartreuse', label='Theoretical Var')

        plt.plot(tics, emp_bias_means, 'b',label='Emperical Bias')
        plt.plot(tics, emp_var, 'g', label='Emperical Var')


        plt.legend()
        plt.savefig("data_analytics/" + filename + "_empbias_var")


        # Plot max/avg distortion
        plt.gcf().clear()

    plt.figure(1)

    # Plot bias/variance graph
    plt.title(filename)
    plt.xlabel('lambda')
    plt.ylabel('normalized units')

    plt.plot(tics, cutoff, 'ro', markersize=2, label='Cutoff')

    plt.plot(tics, theor_bias_samples, 'aqua', label='Theoretical Bias')
    plt.plot(tics, theor_var_samples, 'chartreuse', label='Theoretical Var')


    plt.legend()
    plt.savefig("data_analytics/" +  filename + "_biasvar")

    # Plot max/avg distortion
    plt.gcf().clear()




    plt.figure(2)

    plt.title(filename)
    plt.xlabel('lambda')
    plt.ylabel('normalized units')

    plt.plot(tics, list_estimator_N, 'ro', markersize=2,label='Random Estimator Count')

    plt.plot(tics, max_distortion[:, 0], 'b', label='Max Distortion')
    plt.plot(tics, max_distortion[:, 0] - max_distortion[:, 1], '--b')
    plt.plot(tics, max_distortion[:, 0] + max_distortion[:, 1], '--b')

    plt.plot(tics, p99_distortion[:, 0],  label='p99 Distortion', color='purple')
    plt.plot(tics, p99_distortion[:, 0] - p99_distortion[:, 1], '--', color='purple')
    plt.plot(tics, p99_distortion[:, 0] + p99_distortion[:, 1], '--', color='purple')

    plt.plot(tics, l2_distortion[:, 0], 'g', label='l2 Distortion')
    plt.plot(tics, l2_distortion[:, 0] - l2_distortion[:, 1], '--g')
    plt.plot(tics, l2_distortion[:, 0] + l2_distortion[:, 1], '--g')

    plt.plot(tics, min_max_N[:,0], 'aqua', label="Minimal Naive Max Norm")
    plt.plot(tics, min_l2_N[:,0], 'chartreuse', label="Minimal Naive L2 Norm")

    plt.legend()
    plt.savefig("data_analytics/" + filename + "_distortion")
    plt.gcf().clear()




#used for fmm
def generate_fmm_plots(file_string):
    data = pickle.load(open("data_analytics_fmm/" + file_string + ".p", "rb"))

    print(data.shape)
    #Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0,1)


    #unload data
    [tics, l2_distortion, l2_quick_distortion, l2_rand_distortion] = data[:,:,:]

    tics = tics[:,0]
    plt.figure(1)

    plt.figure(2)

    plt.title(file_string)
    plt.xlabel('target_dim')
    plt.ylabel('log frobenius norm error')


    plt.plot(tics, np.log(l2_rand_distortion[:, 0]), 'b', label='Oblivious Distortion')
    plt.plot(tics, np.log(l2_rand_distortion[:, 0] - l2_rand_distortion[:, 1]), '--b')
    plt.plot(tics, np.log(l2_rand_distortion[:, 0] + l2_rand_distortion[:, 1]), '--b')

    plt.plot(tics, np.log(l2_quick_distortion[:, 0]),  label='Quick Distortion', color='purple')
    plt.plot(tics, np.log(l2_quick_distortion[:, 0] - l2_quick_distortion[:, 1]), '--', color='purple')
    plt.plot(tics, np.log(l2_quick_distortion[:, 0] + l2_quick_distortion[:, 1]), '--', color='purple')

    plt.plot(tics, np.log(l2_distortion[:, 0]), 'g', label='Optimal Distortion')
    plt.plot(tics, np.log(l2_distortion[:, 0] - l2_distortion[:, 1]), '--g')
    plt.plot(tics, np.log(l2_distortion[:, 0] + l2_distortion[:, 1]), '--g')

    plt.legend()
    plt.savefig("data_analytics_fmm/" + file_string + "_distortion")
    plt.gcf().clear()


'''
Table Generators
'''



def generate_regression_tables(filename):
    data = pickle.load(open(filename + ".p", "rb"))

    print(data.shape)
    # Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0,1)

    # unload data
    tics = data[0, :, :]
    lamb_means = data[1:, :, :]

    tics = tics[:, 0]


    lamb_N =lamb_means.shape[0]
    max_N = lamb_N
    for lamb_ind in range(max_N):
        # color_interp = (lamb_ind*1.0)/(max_N-1)
        print("lambda", (2.0*lamb_ind)/(lamb_N-1) -1)
        print("means", lamb_means[lamb_ind, :, 0])
        print("std", lamb_means[lamb_ind, :, 1])





def generate_regression_lamb_tables(filename, increment=50):
    data = pickle.load(open("data_analytics_regress/" + filename + ".p", "rb"))

    # print(data.shape)
    # Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0, 1)

    low_tic = 0
    data = data[:, low_tic:, :]

    # unload data
    tics = data[0, :, :]
    mse = data[1:, :, :]

    tics = tics[:, 0]
    print(tics)
    mse_N = mse.shape[0]

    for mse_ind in range(mse_N):
        # color_interp = (mse_ind*1.0)/mse_N
        print("k",  str(increment*(mse_ind + 1)))
        print("means", mse[mse_ind, :, 0])
        print("std", mse[mse_ind, :, 1])



def generate_interpolation_tables(filename, bias_var = False):
    data = pickle.load(open("data_analytics/" + filename + ".p", "rb"))

    print(data.shape)
    #Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0,1)


    #unload data
    [tics, emp_bias_samples, emp_var_samples, theor_bias_samples, theor_var_samples, max_distortion,
     p99_distortion, l2_distortion, list_estimator_N, cutoff, min_max_N, min_l2_N] = data

    print(emp_bias_samples.shape)


    bias_normalizer = max(theor_bias_samples[:, 0])
    theor_bias_samples = theor_bias_samples[:,0] / bias_normalizer

    var_normalizer = max(theor_var_samples[:, 0])
    theor_var_samples = theor_var_samples[:,0] / var_normalizer

    max_cutoff = max(cutoff[:,0])
    cutoff = cutoff[:,0] / max_cutoff

    normalizer = max(max_distortion[:, 0])
    max_distortion = max_distortion/normalizer
    p99_distortion = p99_distortion/normalizer


    list_estimator_N = list_estimator_N[:,0]/max(list_estimator_N[:,0])

    l2_normalizer = max(l2_distortion[:,0])
    l2_distortion = l2_distortion/l2_normalizer
    # max_distortion = _helper_normalize(max_distortion)
    # avg_distortion = _helper_normalize(avg_distortion)
    # p99_distortion = _helper_normalize(p99_distortion)

    min_max_N = min_max_N/normalizer
    min_l2_N = min_l2_N/l2_normalizer

    tics = tics[:, 0]

    if bias_var:
        emp_bias_means = h._helper_normalize(emp_bias_samples)[:, 0]

        emp_var = h._helper_normalize(emp_var_samples)[:, 0]

        theor_bias_samples = h._helper_normalize(theor_bias_samples)[:, 0]
        theor_var_samples = h._helper_normalize(theor_var_samples)[:, 0]
        # plt.figure(1)

        #Plot bias/variance graph
        # plt.title(filename)
        # plt.xlabel('lambda')
        # plt.ylabel('normalized units')
        print("random estimator count", list_estimator_N)
        # plt.plot(tics, list_estimator_N, 'ro', markersize=2, label='Random Estimator Count')

        print("theoretical bias", theor_bias_samples)
        print("theoretical var", theor_var_samples)

        # plt.plot(tics, theor_bias_samples, 'aqua', label='Theoretical Bias')
        # plt.plot(tics, theor_var_samples, 'chartreuse', label='Theoretical Var')

        print("emperical bias", emp_bias_means)
        print("emperical var", emp_var)

        # plt.plot(tics, emp_bias_means, 'b',label='Emperical Bias')
        # plt.plot(tics, emp_var, 'g', label='Emperical Var')



    plt.figure(1)

    # Plot bias/variance graph
    # plt.title(filename)
    # plt.xlabel('lambda')
    # plt.ylabel('normalized units')

    print("cutoff", cutoff)

    print("theor bias", theor_bias_samples)
    print("theor var", theor_var_samples)





    # plt.plot(tics, list_estimator_N, 'ro', markersize=2,label='Random Estimator Count')
    print("random estimator count", list_estimator_N)

    # plt.plot(tics, max_distortion[:, 0], 'b', label='Max Distortion')
    print("max distortion mean", max_distortion[:, 0] )
    print("max distortion var", max_distortion[:, 1] )
    # plt.plot(tics, max_distortion[:, 0] - max_distortion[:, 1], '--b')
    # plt.plot(tics, max_distortion[:, 0] + max_distortion[:, 1], '--b')

    print("p99 distortion mean", p99_distortion[:, 0])
    print("p99 distortion var", p99_distortion[:, 1])
    # plt.plot(tics, p99_distortion[:, 0],  label='p99 Distortion', color='purple')
    # plt.plot(tics, p99_distortion[:, 0] - p99_distortion[:, 1], '--', color='purple')
    # plt.plot(tics, p99_distortion[:, 0] + p99_distortion[:, 1], '--', color='purple')


    print("l2 distortion mean", l2_distortion[:, 0])
    print("l2 distortion var", l2_distortion[:, 1])
    # plt.plot(tics, l2_distortion[:, 0], 'g', label='l2 Distortion')
    # plt.plot(tics, l2_distortion[:, 0] - l2_distortion[:, 1], '--g')
    # plt.plot(tics, l2_distortion[:, 0] + l2_distortion[:, 1], '--g')

    print("Minimal Naive Max Norm",min_max_N[:,0] )
    print("Minimal Naive L2 Norm",min_l2_N[:,0] )
    # plt.plot(tics, min_max_N[:,0], 'aqua', label="Minimal Naive Max Norm")
    # plt.plot(tics, min_l2_N[:,0], 'chartreuse', label="Minimal Naive L2 Norm")




#used for fmm
def generate_fmm_tables(file_string):
    data = pickle.load(open("data_analytics_fmm/" + file_string + ".p", "rb"))

    # print(data.shape)
    #Changes data so its formated analytics x tics x (bias,var)
    data = np.swapaxes(data, 0,1)


    #unload data
    [tics, l2_distortion, l2_quick_distortion, l2_rand_distortion] = data[:,:,:]

    tics = tics[:,0]
    # plt.figure(1)
    #
    # plt.figure(2)
    #
    # plt.title(file_string)
    # plt.xlabel('target_dim')
    # plt.ylabel('log frobenius norm error')
    print("target_dim", tics)
    print("oblivious distortion", l2_rand_distortion[:, 0])
    print("oblivious distortion var", l2_rand_distortion[:, 1])
    # plt.plot(tics, np.log(l2_rand_distortion[:, 0]), 'b', label='Oblivious Distortion')
    # plt.plot(tics, np.log(l2_rand_distortion[:, 0] - l2_rand_distortion[:, 1]), '--b')
    # plt.plot(tics, np.log(l2_rand_distortion[:, 0] + l2_rand_distortion[:, 1]), '--b')

    print("quick distortion", l2_quick_distortion[:, 0])
    print("quick distortion", l2_quick_distortion[:, 1])
    # plt.plot(tics, np.log(l2_quick_distortion[:, 0]),  label='Quick Distortion', color='purple')
    # plt.plot(tics, np.log(l2_quick_distortion[:, 0] - l2_quick_distortion[:, 1]), '--', color='purple')
    # plt.plot(tics, np.log(l2_quick_distortion[:, 0] + l2_quick_distortion[:, 1]), '--', color='purple')

    print("optimal distortion", l2_distortion[:, 0])
    print("optimal distortion", l2_distortion[:, 1])
    # plt.plot(tics, np.log(l2_distortion[:, 0]), 'g', label='Optimal Distortion')
    # plt.plot(tics, np.log(l2_distortion[:, 0] - l2_distortion[:, 1]), '--g')
    # plt.plot(tics, np.log(l2_distortion[:, 0] + l2_distortion[:, 1]), '--g')


