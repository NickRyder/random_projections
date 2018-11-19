import numpy as np



'''
inputs: 
realizations -      train_N x realizations_N x ambient_dim
target_values -     train_N
test_realzations -  test_N x realizations_N x ambient_dim
test_values -       test_N

outputs: [mean, std]
'''
def regress_realizations_error(realizations, target_values, test_realizations, test_values):
    realizations_N = realizations.shape[1]
    test_error = [0] * realizations_N
    import sklearn.linear_model as sk
    model = sk.LogisticRegression()

    for realization_index in range(realizations_N):
        data_realization = realizations[:,realization_index,:]
        from sklearn import linear_model

        model.fit(data_realization, target_values)
        model_error = model.score(test_realizations[:,realization_index,:], test_values)

        # lstsq = np.linalg.lstsq(data_realization,target_values)
        # regressor = lstsq[0]
        # # print("test_value.shape %s " % str(test_values.shape))
        # # print("test_data.shape %s " % str(test_data.shape))
        # # print("regressor.shape %s" % str(regressor.shape))
        # realization_error_sq = np.linalg.norm(test_values - test_realizations[:, realization_index, :] @ regressor) ** 2
        # print(realization_error_sq- model_error)
        # print("realization_error %s" % str(realization_error))
        test_error[realization_index] = model_error

    return [np.mean(test_error), np.std(test_error)]
