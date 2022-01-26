from __future__ import division
import numpy as np
import os

from scipy.stats import zscore

# Functions to estimate cost for each lambda, by voxel:


from numpy.linalg import inv, svd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import time
from scipy.stats import zscore


def corr(X, Y, axis=0):
    # correlation coefficient
    return np.mean(zscore(X) * zscore(Y), axis)


def R2(Pred, Real):
    # coefficient of determination
    # R^2 = 1 -  residual sum of squares/total sum of squares
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0)
    return np.nan_to_num(1 - SSres / SStot)


def fit_predict(data, features, n_folds=10):
    n, v = data.shape
    p = features.shape[1]
    corrs = np.zeros((n_folds, v))
    R2s = np.zeros((n_folds, v))
    ind = CV_ind(n, n_folds)
    preds_all = np.zeros_like(data)
    for i in range(n_folds):
        train_data = np.nan_to_num(zscore(data[ind != i]))
        train_features = np.nan_to_num(zscore(features[ind != i]))
        test_data = np.nan_to_num(zscore(data[ind == i]))
        test_features = np.nan_to_num(zscore(features[ind == i]))
        weights, __ = cross_val_ridge(train_features, train_data)
        preds = np.dot(test_features, weights)
        preds_all[ind == i] = preds
    #         print("fold {}".format(i))
    corrs = corr(preds_all, data)
    R2s = R2(preds_all, data)
    return corrs, R2s


def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))
    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items :] = n_folds - 1
    return ind


def R2r(Pred, Real):
    # square root of R^2
    R2rs = R2(Pred, Real)
    ind_neg = R2rs < 0  # pick out negative ones
    R2rs = np.abs(R2rs)  # use absolute value to calculate sqaure root
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= -1  # recover negative data
    return R2rs


def ridge(X, Y, lmbda):
    # weight of ridge regression
    return np.dot(inv(X.T.dot(X) + lmbda * np.eye(X.shape[1])), X.T.dot(Y))


def lasso(X, Y, lmbda):
    return soft_ths(ols(X, Y), X.shape[0] * lmbda)


def soft_ths(X, alpha):
    Y = np.zeros_like(X)
    Y[X > alpha] = (X - alpha)[X > alpha]
    Y[X < alpha] = (X + alpha)[X < alpha]

    return Y


# def soft_threshold(alpha, beta):
#     if beta > alpha:
#         return beta - alpha
#     elif beta < -alpha:
#         return beta + alpha
#     else:
#         return 0


def ols(X, Y):
    return np.dot(np.linalg.pinv(X.T.dot(X)), X.T.dot(Y))
    # return np.linalg.inv(X.T @ X) @ (X.T @ Y)


def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    # validation error of ridge regression under different lambdas
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def lasso_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    # validation error of ridge regression under different lambdas
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = lasso(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def ols_err(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros(Y.shape[1])
    weights = ols(X, Y)
    error = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def ridge_sk(X, Y, lmbda):
    rd = Ridge(alpha=lmbda)
    rd.fit(X, Y)
    return rd.coef_.T


def ridgeCV_sk(X, Y, lmbdas):
    rd = RidgeCV(alphas=lmbdas, solver="svd")
    rd.fit(X, Y)
    return rd.coef_.T


def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge_sk(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def ridge_svd(X, Y, lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))


def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx, lmbda in enumerate(lambdas):
        d = s / (s ** 2 + lmbda)
        weights = np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def kernel_ridge(X, Y, lmbda):
    return np.dot(X.T.dot(inv(X.dot(X.T) + lmbda * np.eye(X.shape[0]))), Y)


def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = kernel_ridge(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def kernel_ridge_svd(X, Y, lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return np.dot(np.dot(U, np.diag(d).dot(Vt)), Y)


def kernel_ridge_by_lambda_svd(
    X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])
):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx, lmbda in enumerate(lambdas):
        d = s / (s ** 2 + lmbda)
        weights = np.dot(np.dot(U, np.diag(d).dot(Vt)), Y)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def cross_val_ridge(
    train_features,
    train_data,
    n_splits=10,
    lambdas=np.array([10 ** i for i in range(-6, 10)]),
    method="plain",
    do_plot=False,
):
    # cross validation for ridge regression

    ridge_1 = dict(
        plain=ridge_by_lambda,
        svd=ridge_by_lambda_svd,
        kernel_ridge=kernel_ridge_by_lambda,
        kernel_ridge_svd=kernel_ridge_by_lambda_svd,
        ridge_sk=ridge_by_lambda_sk,
    )[
        method
    ]  # loss of the regressor
    ridge_2 = dict(
        plain=ridge,
        svd=ridge_svd,
        kernel_ridge=kernel_ridge,
        kernel_ridge_svd=kernel_ridge_svd,
        ridge_sk=ridge_sk,
    )[
        method
    ]  # solver for the weights

    n_voxels = train_data.shape[1]  # get number of voxels from data
    nL = lambdas.shape[0]  # get number of hyperparameter (lambdas) from setting
    r_cv = np.zeros((nL, train_data.shape[1]))  # loss matrix

    kf = KFold(n_splits=n_splits)  # set up dataset for cross validation
    start_t = time.time()  # record start time
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        # print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(
            train_features[trn],
            train_data[trn],
            train_features[val],
            train_data[val],
            lambdas=lambdas,
        )  # loss of regressor 1
        if do_plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(cost, aspect="auto")
        r_cv += cost
    #         if icv%3 ==0:
    #             print(icv)
    #         print('average iteration length {}'.format((time.time()-start_t)/(icv+1))) # time used
    if do_plot:  # show loss
        plt.figure()
        plt.imshow(r_cv, aspect="auto", cmap="RdBu_r")

    argmin_lambda = np.argmin(r_cv, axis=0)  # pick the best lambda
    weights = np.zeros(
        (train_features.shape[1], train_data.shape[1])
    )  # initialize the weight
    for idx_lambda in range(
        lambdas.shape[0]
    ):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:, idx_vox] = ridge_2(
            train_features, train_data[:, idx_vox], lambdas[idx_lambda]
        )
    if do_plot:  # show the weights
        plt.figure()
        plt.imshow(weights, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])


def cross_val_lasso(
    train_features,
    train_data,
    n_splits=10,
    lambdas=np.array([10 ** i for i in range(-6, 10)]),
    method="plain",
    do_plot=False,
):
    # cross validation for ridge regression

    # ridge_1 = dict(plain = ridge_by_lambda,
    #                svd = ridge_by_lambda_svd,
    #                kernel_ridge = kernel_ridge_by_lambda,
    #                kernel_ridge_svd = kernel_ridge_by_lambda_svd,
    #                ridge_sk = ridge_by_lambda_sk)[method] #loss of the regressor
    # ridge_2 = dict(plain = ridge,
    #                svd = ridge_svd,
    #                kernel_ridge = kernel_ridge,
    #                kernel_ridge_svd = kernel_ridge_svd,
    #                ridge_sk = ridge_sk)[method] # solver for the weights

    n_voxels = train_data.shape[1]  # get number of voxels from data
    nL = lambdas.shape[0]  # get number of hyperparameter (lambdas) from setting
    r_cv = np.zeros((nL, train_data.shape[1]))  # loss matrix

    kf = KFold(n_splits=n_splits)  # set up dataset for cross validation
    start_t = time.time()  # record start time
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        print("ntrain = {}".format(train_features[trn].shape[0]))
        cost = lasso_by_lambda(
            train_features[trn],
            train_data[trn],
            train_features[val],
            train_data[val],
            lambdas=lambdas,
        )  # loss of regressor 1
        if do_plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(cost, aspect="auto")
        r_cv += cost
        if icv % 3 == 0:
            print(icv)
        print(
            "average iteration length {}".format((time.time() - start_t) / (icv + 1))
        )  # time used
    if do_plot:  # show loss
        plt.figure()
        plt.imshow(r_cv, aspect="auto", cmap="RdBu_r")

    argmin_lambda = np.argmin(r_cv, axis=0)  # pick the best lambda
    weights = np.zeros(
        (train_features.shape[1], train_data.shape[1])
    )  # initialize the weight
    for idx_lambda in range(
        lambdas.shape[0]
    ):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:, idx_vox] = lasso(
            train_features, train_data[:, idx_vox], lambdas[idx_lambda]
        )
    if do_plot:  # show the weights
        plt.figure()
        plt.imshow(weights, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])


def cross_val_ols(
    train_features,
    train_data,
    n_splits=10,
    lambdas=np.array([10 ** i for i in range(-6, 10)]),
    method="plain",
    do_plot=False,
):
    # cross validation for ridge regression

    # ridge_1 = dict(plain = ridge_by_lambda,
    #                svd = ridge_by_lambda_svd,
    #                kernel_ridge = kernel_ridge_by_lambda,
    #                kernel_ridge_svd = kernel_ridge_by_lambda_svd,
    #                ridge_sk = ridge_by_lambda_sk)[method] #loss of the regressor
    # ridge_2 = dict(plain = ridge,
    #                svd = ridge_svd,
    #                kernel_ridge = kernel_ridge,
    #                kernel_ridge_svd = kernel_ridge_svd,
    #                ridge_sk = ridge_sk)[method] # solver for the weights

    n_voxels = train_data.shape[1]  # get number of voxels from data
    nL = lambdas.shape[0]  # get number of hyperparameter (lambdas) from setting
    r_cv = np.zeros((nL, train_data.shape[1]))  # loss matrix

    kf = KFold(n_splits=n_splits)  # set up dataset for cross validation
    start_t = time.time()  # record start time
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        print("ntrain = {}".format(train_features[trn].shape[0]))
        cost = ols_err(
            train_features[trn],
            train_data[trn],
            train_features[val],
            train_data[val],
            lambdas=lambdas,
        )  # loss of regressor 1
        if do_plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(cost, aspect="auto")
        r_cv += cost
        if icv % 3 == 0:
            print(icv)
        print(
            "average iteration length {}".format((time.time() - start_t) / (icv + 1))
        )  # time used
    if do_plot:  # show loss
        plt.figure()
        plt.imshow(r_cv, aspect="auto", cmap="RdBu_r")

    argmin_lambda = np.argmin(r_cv, axis=0)  # pick the best lambda
    weights = np.zeros(
        (train_features.shape[1], train_data.shape[1])
    )  # initialize the weight
    for idx_lambda in range(
        lambdas.shape[0]
    ):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:, idx_vox] = ols(train_features, train_data[:, idx_vox])
    if do_plot:  # show the weights
        plt.figure()
        plt.imshow(weights, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])


def GCV_ridge(
    train_features, train_data, lambdas=np.array([10 ** i for i in range(-6, 10)])
):

    n_lambdas = lambdas.shape[0]
    n_voxels = train_data.shape[1]
    n_time = train_data.shape[0]
    n_p = train_features.shape[1]

    CVerr = np.zeros((n_lambdas, n_voxels))

    # % If we do an eigendecomp first we can quickly compute the inverse for many different values
    # % of lambda. SVD uses X = UDV' form.
    # % First compute K0 = (X'X + lambda*I) where lambda = 0.
    # K0 = np.dot(train_features,train_features.T)
    print(
        "Running svd",
    )
    start_time = time.time()
    [U, D, Vt] = svd(train_features, full_matrices=False)
    V = Vt.T
    print(U.shape, D.shape, Vt.shape)
    print("svd time: {}".format(time.time() - start_time))

    for i, regularizationParam in enumerate(lambdas):
        regularizationParam = lambdas[i]
        print("CVLoop: Testing regularization param: {}".format(regularizationParam))

        # Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
        dlambda = D ** 2 + np.eye(n_p) * regularizationParam
        dlambdaInv = np.diag(D / np.diag(dlambda))
        KlambdaInv = V.dot(dlambdaInv).dot(U.T)

        # Compute S matrix of Hastie Trick  H = X(XT X + lambdaI)-1XT
        S = np.dot(U, np.diag(D * np.diag(dlambdaInv))).dot(U.T)
        denum = 1 - np.trace(S) / n_time

        # Solve for weight matrix so we can compute residual
        weightMatrix = KlambdaInv.dot(train_data)

        #         Snorm = np.tile(1 - np.diag(S) , (n_voxels, 1)).T
        YdiffMat = train_data - (train_features.dot(weightMatrix))
        YdiffMat = YdiffMat / denum
        CVerr[i, :] = (1 / n_time) * np.sum(YdiffMat * YdiffMat, 0)

    # try using min of avg err
    minerrIndex = np.argmin(CVerr, axis=0)
    r = np.zeros((n_voxels))

    for nPar, regularizationParam in enumerate(lambdas):
        ind = np.where(minerrIndex == nPar)[0]
        if len(ind) > 0:
            r[ind] = regularizationParam
            print(
                "{}% of outputs with regularization param: {}".format(
                    int(len(ind) / n_voxels * 100), regularizationParam
                )
            )
            # got good param, now obtain weights
            dlambda = D ** 2 + np.eye(n_p) * regularizationParam
            dlambdaInv = np.diag(D / np.diag(dlambda))
            KlambdaInv = V.dot(dlambdaInv).dot(U.T)

            weightMatrix[:, ind] = KlambdaInv.dot(train_data[:, ind])

    return weightMatrix, r


score_f = R2
# score_f = corr
# score_f = cosine


def CV_ind(n, n_folds):
    # index for cross validation
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))  # number of items in one fold
    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items :] = n_folds - 1
    return ind


def stacked_core(
    n_voxels,
    feat_use,
    err,
    train_data,
    preds_test,
    preds_train,
    test_ind,
    ind_num,
    stacked_pred,
    stacked_train_r2s_fold,
    S_average,
):

    n_features = len(feat_use)
    # calculate error matrix for stacking
    P = np.zeros((n_voxels, n_features, n_features))
    idI = 0
    for i in feat_use:
        idJ = 0
        for j in feat_use:
            P[:, idI, idJ] = np.mean(err[i] * err[j], 0)
            idJ += 1
        idI += 1

    idI = 0
    idJ = 0

    # PROGRAMATICALLY SET THIS FROM THE NUMBER OF FEATURES
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))

    stacked_pred_train = np.zeros_like(train_data)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        # solve for stacking weights for every voxel
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(
            n_features,
        )
        # combine the prednp.unique(np.nonzero(dt)[1]).shapeictions from the individual feature spaces for voxel i
        z = np.array([preds_test[feature_j, test_ind, i] for feature_j in feat_use])
        if i == 0:
            print(z.shape)  # to make sure
        # multiply the predictions by S[i,:]
        stacked_pred[test_ind, i] = np.dot(S[i, :], z)
        # combine the training predictions from the individual feature spaces for voxel i
        z = np.array([preds_train[feature_j][:, i] for feature_j in feat_use])
        stacked_pred_train[:, i] = np.dot(S[i, :], z)
    #             if score_f(stacked_pred_train[:,i],train_data[:,i])>0.1:
    #                 a = blablabla

    S_average += S

    # stacked prediction, computed over a fold
    # stacked_r2s_fold[ind_num,:] = score_f(stacked_pred[test_ind],test_data)

    stacked_train_r2s_fold[ind_num, :] = score_f(stacked_pred_train, train_data)

    return stacked_pred, stacked_train_r2s_fold, S_average, S


# THIS IS HOW I WOULD RUN A REGRESSION ANALYSIS: VERY VERY IMPORTANT
# FMRI IS SLOW IN TIME AND WE CANNOT SAMPLE RANDOM POINTS... INSTEAD
# WE HAVE TO USE CONTIGUOUS BLOCKS

from cvxopt import matrix, solvers

solvers.options["show_progress"] = False

score_f = R2
# score_f = corr
# score_f = cosine


def CV_ind(n, n_folds):
    # index for cross validation
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))  # number of items in one fold
    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items :] = n_folds - 1
    return ind


def stacking_CV_fmri(data, features, method="simple_ridge", n_folds=4):

    # INPUTS: data (ntime*nvoxels), features (list of ntime*ndim), method = what to use to train,
    #         n_folds = number of cross-val folds

    n_time = data.shape[0]
    n_voxels = data.shape[1]
    n_features = len(features)

    ind = CV_ind(n_time, n_folds=n_folds)

    # easier to store r2s in an array and access them programatically than to maintain a different
    # variable for each
    r2s = np.zeros((n_features, n_voxels))
    # r2s_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    # r2s_weighted_fold = np.zeros((n_folds, n_features, n_voxels))
    # stacked_r2s_fold = np.zeros((n_folds, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))

    # store predictions in array
    stacked_pred = np.zeros((n_time, n_voxels))
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_features, n_time, n_voxels))

    #
    S_average = np.zeros((n_voxels, n_features))

    stacked_pred_lo = dict()
    stacked_train_r2s_fold_lo = dict()
    S_average_lo = dict()

    for t in range(n_features):
        stacked_pred_lo[t] = np.zeros((n_time, n_voxels))
        stacked_train_r2s_fold_lo[t] = np.zeros((n_folds, n_voxels))
        S_average_lo[t] = np.zeros((n_voxels, n_features - 1))

    # DO BY FOLD
    for ind_num in range(n_folds):
        train_ind = ind != ind_num
        test_ind = ind == ind_num

        # split data
        train_data = data[train_ind]
        train_features = [F[train_ind] for F in features]

        test_data = data[test_ind]
        test_features = [F[test_ind] for F in features]

        # normalize data  <= WE SHOULD ZSCORE BY TRAIN/TEST
        train_data = np.nan_to_num(zscore(train_data))
        test_data = np.nan_to_num(zscore(test_data))

        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]

        err = dict()
        preds_train = dict()

        #         for FEATURE in range(n_features):
        #             preds_train[FEATURE], error, preds_test[FEATURE,test_ind], r2s_train_folds[ind_num,FEATURE,:], _ = feat_ridge_CV(train_features[FEATURE],
        #                                                                train_data,
        #                                                                test_features[FEATURE],method=method)
        #             err[FEATURE] = error

        for FEATURE in range(n_features):
            if method == "simple_ridge":
                weights = ridge(train_features[FEATURE], train_data, 100)
            elif method == "cross_val_ridge":
                weights, __ = cross_val_ridge(
                    train_features[FEATURE],
                    train_data,
                    n_splits=4,
                    lambdas=np.array([10 ** i for i in range(-6, 10)]),
                    do_plot=False,
                )
            preds_train[FEATURE] = np.dot(train_features[FEATURE], weights)
            err[FEATURE] = train_data - preds_train[FEATURE]
            # predict the test data also before overwriting the weights:
            preds_test[FEATURE, test_ind] = np.dot(test_features[FEATURE], weights)
            # preds_test[FEATURE,test_ind] = zscore(preds_test[FEATURE][test_ind])
            # single feature space predictions, computed over a fold
            # r2s_folds[ind_num,FEATURE,:] = score_f(preds_test[FEATURE,test_ind],test_data)
            r2s_train_folds[ind_num, FEATURE, :] = score_f(
                preds_train[FEATURE], train_data
            )

        stacked_pred, stacked_train_r2s_fold, S_average, S = stacked_core(
            n_voxels,
            range(n_features),
            err,
            train_data,
            preds_test,
            preds_train,
            test_ind,
            ind_num,
            stacked_pred,
            stacked_train_r2s_fold,
            S_average,
        )

        for leave_one in range(n_features):
            feat_use = list(range(n_features))
            feat_use.remove(leave_one)
            (
                stacked_pred_lo[leave_one],
                stacked_train_r2s_fold_lo[leave_one],
                S_average_lo[leave_one],
                _,
            ) = stacked_core(
                n_voxels,
                feat_use,
                err,
                train_data,
                preds_test,
                preds_train,
                test_ind,
                ind_num,
                stacked_pred_lo[leave_one],
                stacked_train_r2s_fold_lo[leave_one],
                S_average_lo[leave_one],
            )

        for FEATURE in range(n_features):
            # weight the predictions according to S:
            # weighted single feature space predictions, computed over a fold
            weighted_pred[FEATURE, test_ind] = (
                preds_test[FEATURE, test_ind] * S[:, FEATURE]
            )
            # r2s_weighted_fold[ind_num,FEATURE,:] = score_f(weighted_pred[FEATURE,test_ind],test_data)

    # compute overall
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], data)
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], data)

    stacked_r2s = score_f(stacked_pred, data)

    stacked_r2s_lo = np.zeros((n_features, n_voxels))
    for FEATURE in range(n_features):
        stacked_r2s_lo[FEATURE, :] = score_f(stacked_pred_lo[FEATURE], data)
        S_average_lo[FEATURE] = S_average_lo[FEATURE] / n_folds

    r2s_train = r2s_train_folds.mean(0)
    stacked_train = stacked_train_r2s_fold.mean(0)

    S_average = S_average / n_folds

    return (
        r2s,
        stacked_r2s,
        stacked_r2s_lo,
        r2s_weighted,
        r2s_train,
        stacked_train,
        S_average,
        S_average_lo,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    alexnet_conv1 = np.load(
        "/user_data/ruogulin/NSD_subj1_lenet5_cifar10_sparse_095/cifar10_lenet5_conv1.npy"
    ).squeeze()
    alexnet_conv2 = np.load(
        "/user_data/ruogulin/NSD_subj1_lenet5_cifar10_sparse_095/cifar10_lenet5_conv2.npy"
    ).squeeze()
    alexnet_conv3 = np.load(
        "/user_data/ruogulin/NSD_subj1_lenet5_cifar10_sparse_095/cifar10_lenet5_fc3.npy"
    ).squeeze()
    alexnet_conv4 = np.load(
        "/user_data/ruogulin/NSD_subj1_lenet5_cifar10_sparse_095/cifar10_lenet5_fc4.npy"
    ).squeeze()
    alexnet_conv5 = np.load(
        "/user_data/ruogulin/NSD_subj1_alexnetB_V1d_sparse_095/NSD_alexnet_conv5.npy"
    ).squeeze()
    alexnet_fc6 = np.load(
        "/user_data/ruogulin/NSD_subj1_alexnetB_V1d_sparse_095/NSD_alexnet_fc6.npy"
    ).squeeze()
    alexnet_fc7 = np.load(
        "/user_data/ruogulin/NSD_subj1_alexnetB_V1d_sparse_095/NSD_alexnet_fc7.npy"
    ).squeeze()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=512)
    alexnet_conv1 = pca.fit_transform(alexnet_conv1)
    alexnet_conv2 = pca.fit_transform(alexnet_conv2)
    alexnet_conv3 = pca.fit_transform(alexnet_conv3)
    alexnet_conv4 = pca.fit_transform(alexnet_conv4)
    alexnet_conv5 = pca.fit_transform(alexnet_conv5)
    alexnet_fc6 = pca.fit_transform(alexnet_fc6)
    alexnet_fc7 = pca.fit_transform(alexnet_fc7)

    dt = np.load(
        "/home/ruogulin/NSD_subj1_alexnet_imagenet/averaged_cortical_responses_zscored_by_run_subj01.npy"
    )

    feature_list = []

    feature_list.append(alexnet_conv1)
    feature_list.append(alexnet_conv2)
    feature_list.append(alexnet_conv3)
    feature_list.append(alexnet_conv4)
    feature_list.append(alexnet_conv5)
    feature_list.append(alexnet_fc6)
    feature_list.append(alexnet_fc7)

    import pickle

    split = 5000
    k = dt.shape[1] // split
    t = k * split

    for i in range(0, k + 1):
        if i * split < t:
            j = (i + 1) * split
        else:
            j = dt.shape[1]
        dt_1 = dt[:, i * split : j]
        (
            r2s,
            stacked_r2s,
            stacked_r2s_lo,
            r2s_weighted,
            r2s_train,
            stacked_train,
            S_average,
            S_average_lo,
        ) = stacking_CV_fmri(dt_1, feature_list, method="cross_val_ridge", n_folds=5)

        part = str(i + 1)

        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/r2s_" + part + ".npy", r2s)
        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/stacked_r2s_" + part + ".npy", stacked_r2s)
        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/stacked_r2s_lo_" + part + ".npy", stacked_r2s_lo)
        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/r2s_w_" + part + ".npy", r2s_weighted)
        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/r2s_train" + part + ".npy", r2s_train)
        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/stacked_train_" + part + ".npy", stacked_train)
        np.save("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/S_" + part + ".npy", S_average)
        f = open("/user_data/ruogulin/subj1_stack_lenet5_cifar10_sparse_095/S_lo_" + part + ".pkl", "wb")
        pickle.dump(S_average_lo, f)
        f.close()
