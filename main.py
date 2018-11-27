import numpy as np
import sys


#TODO: write verifications for projections
#TODO: inputs sometimes data, estimator, data estimator ---- data, data, estimator, estimator

#TODO: have bad habit of intializing lists using [0] * N, this unintentionally typecasts


if __name__ == '__main__':
    np.set_printoptions(linewidth=300)

    n = 100
    k = 3
    Delta = 101
    samplesN = 1000
    realizationsN = 20

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
        samplesN = int(sys.argv[2])
        realizationsN = int(sys.argv[3])

    # Synthetic FMM
    # covariance_X = h.randomPSD(n)
    # covariance_W = h.randomPSD(n)

    import plotter
    print("BIOLOGICAL---------------------------------------")
    print("arcene_700_5000_100")
    plotter.generate_fmm_tables("arcene_700_5000_100")
    print("arcene_5000_700_100")
    plotter.generate_fmm_tables("arcene_5000_700_100")
    print("isolet_308_1559_100")
    plotter.generate_fmm_tables("isolet_308_1559_100")
    print("isolet_t_1559_308_100")
    plotter.generate_fmm_tables("isolet_t_1559_308_100")
    print("-")
    print("-")
    print("-")
    print("-")
    print("-")
    print("SPARSE---------------------------------------")
    print("go_sf_3853_2593_100")
    plotter.generate_fmm_tables("go_sf_3853_2593_100")
    print("go_sf_t_2593_3853_100")
    plotter.generate_fmm_tables("go_sf_t_2593_3853_100")
    print("tw_oc_5673_5000_100")
    plotter.generate_fmm_tables("tw_oc_5673_5000_100")
    print("tw_oc_t_5000_5673_100")
    plotter.generate_fmm_tables("tw_oc_t_5000_5673_100")
    print("-")
    print("-")
    print("-")
    print("-")
    print("-")
    print("LINEAR REGRESSION-------------------------------")
    print("slice_localization_384_53500_5_lamb_5")
    plotter.generate_regression_lamb_tables("slice_localization_384_53500_5_lamb_5",increment=5)
    print("e2006_full_150360_16087_20_lamb_50")
    plotter.generate_regression_lamb_tables("e2006_full_150360_16087_20_lamb_50")

    print("-")
    print("-")
    print("-")
    print("-")
    print("-")
    print("LINEAR REGRESSION-------------------------------")
    print("cifar_logistic_3072_1979_100_lamb_50")
    plotter.generate_regression_lamb_tables("cifar_logistic_3072_1979_100_lamb_50")
    print("rcv1_47236_20242_15_lamb_50")
    plotter.generate_regression_lamb_tables("rcv1_47236_20242_15_lamb_50")


    # Grab slice locaization data
    # data = ds.import_slice_localization()
    # np.random.shuffle(data)
    # samples_N = data.shape[0]
    # ambient_dim = data.shape[1] - 1
    # test_frac = .1
    #
    # data_regress = data[:,:-1]
    # data_values = data[:,-1]
    #
    # ambient_dim = data_regress.shape[1]
    # samples_N = data_regress.shape[0]
    # realizations_N = 100

    import numpy.random as npr





    # import time
    # N=10000
    # n=10000
    #
    # tic = time.time()
    # print(pc.create_rand_estimator(N, n, True).shape)
    # print(time.time() - tic)
    #
    #
    # tic = time.time()
    # print(pc.create_rand_estimator(N, n, False).shape)
    # print(time.time()-tic)


    # data_regress, data_values = ds.import_libsvm("rcv1_train.binary")






    #
    # regress,values = ds.import_libsvm("E2006.train")
    # print(regress[1])
    # data_regress, data_values = ds.import_libsvm("cifar10")
    # data_regress = data_regress[:10000,:].toarray()
    # data_values = data_values[:10000]


    #
    # print("gathered, trimming...")
    # data_regress, data_values = h.trim_classification(data_regress, data_values)
    #
    # print("trimmed")
    # da.generate_save_plot_regression_analytics_lamb(data_regress,data_values,"cifar_logistic",target_dim=50)


    # da.generate_save_plot_regression_analytics_lamb(data_regress,data_values,"rcv1", target_dim=50, realizations_N=15)
    # import plotter
    # plotter.generate_regression_lamb_plots("slice_localization_384_53500_5_lamb_5")








    # # split into test and train
    # test_samples_N = np.int(np.floor(samples_N * test_frac))
    # test_data = data_regress[:test_samples_N, :]
    # test_data_values = data_values[:test_samples_N]
    # train_data = data_regress[test_samples_N:, :]
    # train_data_values = data_values[test_samples_N:]
    #
    # # calculate covariance of train
    # train_data = da.center_data(train_data)
    # cov = da.emperical_covariance(train_data)
    # cov_diag = np.diag(cov)
    #
    # lamb = 0
    # est = (np.diag(h.psuedo_power(cov_diag, lamb)), np.array([], dtype=np.int64).reshape(0, ambient_dim), k)
    #
    # # project all of the data
    # realizations_X, realizations_W = pc.create_random_projection_realizations(data_regress, est,
    #                                                                           data_regress, est,
    #                                                                           realizations_N)
    # train_realizations = realizations_X[test_samples_N:, :, :]
    # test_realizations = realizations_X[:test_samples_N, :, :]
    # print(str(regressors.regress_realizations_error(train_realizations, train_data_values, test_realizations, test_data_values)))
    #
    # est = (.00000001 * np.identity(len(cov_diag)), np.array([], dtype=np.int64).reshape(0, ambient_dim), k)
    #
    # # project all of the data
    # realizations_X, realizations_W = pc.create_random_projection_realizations(data_regress, est,
    #                                                                           data_regress, est,
    #                                                                           realizations_N)
    # train_realizations = realizations_X[test_samples_N:, :, :]
    # test_realizations = realizations_X[:test_samples_N, :, :]
    # print(str(regressors.regress_realizations_error(train_realizations, train_data_values, test_realizations, test_data_values)))



    # import plotter
    # plotter.generate_regression_plots("slice_localization_384_53500_100")

    #split into test and train
    # test_frac = .1
    # test_samples_N = np.int(np.floor(samples_N*test_frac))
    # test_data = data_regress[:test_samples_N, :]
    # test_data_values = data_values[:test_samples_N]
    # train_data = data_regress[test_samples_N:,:]
    # train_data_values = data_values[test_samples_N:]
    #
    # #calculate covariance of train
    # train_data = da.center_data(train_data)
    # cov = da.emperical_covariance(train_data)
    # cov_diag = np.diag(cov)
    # k = np.int(np.floor(np.log(ambient_dim)))
    # lamb = .25
    # est = (np.diag(h.psuedo_power(cov_diag, lamb)), np.array([], dtype=np.int64).reshape(0, ambient_dim), k)
    # print(str(np.diag(h.psuedo_power(cov_diag, lamb)).shape))
    # print(str(ambient_dim))
    # #project all of the data
    #
    # realizations_X, realizations_W = pc.create_random_projection_realizations(data[:,:-1], est, data[:, :-1], est, 100)
    # train_realizations_X = realizations_X[test_samples_N:, :, :]
    # test_realizations_X = realizations_X[:test_samples_N, :, :]
    # print(regressors.regress_realizations_error(train_realizations_X, train_data_values, test_realizations_X, test_data_values))









    # da.generate_save_plot_fmm_analytics(data_X, data_W, "arcene")
    # data_X = np.transpose(data_X)
    # data_W = np.transpose(data_W)
    # da.generate_save_plot_fmm_analytics(data_X, data_W, "arcene")
    #
    # data = ds.import_repeat_consumption("reddit_sample")
    # print(data.shape)
    # attribute_N = data.shape[1]
    # attr_half = int(np.floor(attribute_N/2))
    # data_X = data[:5000, :attr_half]
    # data_W = data[:5000, -attr_half:]
    # da.generate_save_plot_fmm_analytics(data_X, data_W, "reddit_sample")
    # data_X = np.transpose(data_X)
    # data_W = np.transpose(data_W)
    # da.generate_save_plot_fmm_analytics(data_X, data_W, "reddit_sample")

    # import ProjectionComputer
    # ProjectionComputer.test_create_random_projection_realizations()


    # isolet = ds.draw_isolet()
    # data_X = isolet[:, :308]
    # data_W = isolet[:, -308:]
    # generate_save_plot_fmm_analytics(data_X, data_W, "isolet")
    # data_X = np.transpose(data_X)
    # data_W = np.transpose(data_W)
    # generate_save_plot_fmm_analytics(data_X, data_W, "isolet_t")

    # np.diag(np.abs(npr.laplace(size=n)))
    # np.diag(np.flip(np.abs(npr.laplace(size=n))),0))

    # n = 100
    # k = 50
    # covariance_X = h.randomPSD(n)
    # # covariance_X += np.diag(np.arange(n)) * (np.sum(np.diag(covariance_X)) / np.sum(np.arange(n))) * n
    # # covariance_W = h.randomPSD(n)
    # data_X = ds.generateNSamples(covariance_X, samplesN)
    # data_W = ds.generateNSamples(covariance_W, samplesN)
    #
    # mnist_data = ds.draw_MNIST(1000)
    # # for k_exp in range(7,10):
    # k = 128
    # da.generate_save_plot_interpolation_analytics(mnist_data, k, "mnist", realizations_N=20)
    # da.generate_naive_interp_plots(mnist_data,  "mnist", k,realizations_N=20)

    # da.generate_naive_plots(mnist_data, "mnist", k, realizations_N=5)

    # cifar_data = ds.draw_cifar(5000)
    #
    # n=100
    #
    # covariance_X = h.randomPSD(n)
    # # # covariance_X += np.diag(np.arange(n)) * (np.sum(np.diag(covariance_X)) / np.sum(np.arange(n))) * n
    # # covariance_W = h.randomPSD(n)
    # data_X = ds.generateNSamples(covariance_X, samplesN)
    # # for k_exp in range(7,10):
    # #     k = 2 ** k_exp
    # #     da.generate_naive_interp_plots(cifar_data, "cifar_data", k)
    #
    # da.generate_naive_interp_plots(data_X, "test", 50)


    # for k_exp in range(7,10):
    #     k = 2 ** k_exp
    #
    #     import plotter
    #     file_name = da.interpolation_filestring("mnist", k, 1000, 20)
    #     plotter.generate_interpolation_plots(file_name)
        # da.generate_save_plot_interpolation_analytics(cifar_data, k, "cifar", realizations_N = 20)
        # da.generate_naive_plots(cifar_data, "cifar", k, realizations_N=5)


    # import plotter
    # plotter.generate_interpolation_plots("test_interp_3_1000_100")
    # preprocess = orp.preprocess_optimal_mixed_estimators(covariance_X,covariance_W)
    # estimator_X, estimator_W, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(preprocess, k , .2)
    # output_X, output_W = pc.create_random_projection_realizations(data_X, estimator_X, data_W, estimator_W, 2)
    # for i in range(len(estimator_X)):
    #     print(estimator_X[i] - estimator_W[i])
    # print(output_X-output_W)


    # da.generate_save_plot_fmm_analytics(data_X, data_W, "rand_randskew")

    # covariance_X = h.randomPSD(n)
    # # covariance_X += np.diag(np.arange(n)) * (np.sum(np.diag(covariance_X)) / np.sum(np.arange(n))) * n
    # covariance_W = h.randomPSD(n)
    # data_X = ds.generateNSamples(covariance_X, samplesN)
    # data_W = ds.generateNSamples(covariance_W, samplesN)
    # generate_save_plot_fmm_analytics(data_X, data_W, "rand_rand")
    #
    # covariance_X = np.diag(np.arange(n))
    # # covariance_X += np.diag(np.arange(n)) * (np.sum(np.diag(covariance_X)) / np.sum(np.arange(n))) * n
    # covariance_W = h.randomPSD(n)
    # data_X = ds.generateNSamples(covariance_X, samplesN)
    # data_W = ds.generateNSamples(covariance_W, samplesN)
    # generate_save_plot_fmm_analytics(data_X, data_W, "rand_diag")
    #
    #
    # covariance_X = np.diag(np.abs(npr.laplace(size = n)))
    # covariance_W = np.diag(np.flip(np.abs(npr.laplace(size = n)), 0))
    # data_X = ds.generateNSamples(covariance_X, samplesN)
    # data_W = ds.generateNSamples(covariance_W, samplesN)
    # generate_save_plot_fmm_analytics(data_X, data_W, "diag_diag")

    # #center data
    # data_X = da.center_data(data_X)
    # data_W = da.center_data(data_W)
    #
    # #compute emperical covariance
    # emp_covariance_X = da.empericalCovariance(data_X)
    # emp_covariance_W = da.empericalCovariance(data_W)
    #
    # preprocess = orp.preprocess_optimal_mixed_estimators(covariance_X, covariance_W)
    # preprocess_quick = orp.preprocess_quick_mixed_estimators(np.diag(covariance_X), np.diag(covariance_W))
    #
    # estimator_X, estimator_W, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(preprocess, k, -1)
    # estimator_quick_X, estimator_quick_W, cutoff, obj_bias, obj_var = orp.optimal_mixed_estimators(preprocess_quick, k, -1)
    # estimator_random = orp._all_random_estimator_tuple(n, k)
    #
    # print(estimator_random[0])
    # print(estimator_quick_X[0])
    # print(estimator_X[0])
    # print(covariance_X)
    # print(emp_covariance_X)



    # mix_N = 3

    # Sampling
    # orp.test_preprocess_optimal_mixed_estimators()

    # samplesN = 2000
    # realizationsN = 100


    # k = 256
    # drawSamples = ds.draw_cifar(samplesN)
    # generate_analytics_all(drawSamples, k, "cifar", samplesN, realizationsN, Delta, bias_var=False)
    # k = 512
    # drawSamples = ds.draw_cifar(samplesN)
    # generate_analytics_all(drawSamples, k, "cifar", samplesN, realizationsN, Delta, bias_var=False)
    # k = 1024
    # drawSamples = ds.draw_cifar(samplesN)
    # generate_analytics_all(drawSamples, k, "cifar", samplesN, realizationsN, Delta, bias_var=False)
    # # Mixed Guassian Sampling
    # covariance_list = [h.randomPSD(n) for i in range(mix_N)]
    # mean_list = [npr.rand(n) for i in range(mix_N)]
    # distribution = npr.rand(mix_N)
    # distribution = distribution / np.sum(distribution)
    # drawSamples = ds.generateSamplesMixed(covariance_list, mean_list, distribution, samplesN)

    # Testing lambda = 0 for _mixedObjectived_sortediagonalSolver