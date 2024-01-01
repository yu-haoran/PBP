import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Masking, GRU, LSTM
from decimal import Decimal
from sklearn.metrics import f1_score
import random
from scipy.stats import truncnorm
from tensorflow.keras import backend as K
import pingouin as pg
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def masked_mse(yc_actual, yc_predict):
    mask = K.cast(K.not_equal(yc_actual, -1), K.floatx())
    mask_length = K.cast(K.not_equal(yc_actual, -1000), K.floatx())
    out1 = tf.keras.losses.mean_squared_error(yc_actual * mask, yc_predict * mask) * K.sum(mask_length) / K.max([K.sum(mask), 0.1])
    return out1
# Reason of adding K.max([, ]) at the end: if all the data in a batch happens to have no reasonable y_c, then the
# K.sum(mask) will be zero and an error will appear since dividing by zero is not allowed. After adding K.max([, ]),
# if K.sum(mask) is not zero (i.e., >= one), nothing changes; if K.sum(mask) is zero, the numerator is zero and hence
# out1 is still zero and no error will appear.

def findrecentask(xpre_value):
    if np.isnan(xpre_value[14]) != 1:
        brecent = xpre_value[15]
        srecent = xpre_value[14]
    else:
        if np.isnan(xpre_value[8]) != 1:
            brecent = xpre_value[9]
            srecent = xpre_value[8]
        else:
            brecent = xpre_value[3]
            srecent = xpre_value[2]
    return brecent, srecent

def calculatemse(yc_test, yc_pred):
    sum = 0
    countnonzero = 0
    for int in range(len(yc_test)):
        if yc_test[int] != -1:  # only consider the instance with a meaningful counter price value
            countnonzero = countnonzero + 1
            sum = sum + (yc_test[int] - yc_pred[int]) ** 2
    return sum/countnonzero

def classifytestdata(x_test, yd_test, yc_test):
    xlength = len(x_test[0, :])
    x_test1 = np.empty((0, xlength))
    x_test2 = np.empty((0, xlength))
    x_test3 = np.empty((0, xlength))
    yd_test1 = []
    yd_test2 = []
    yd_test3 = []
    yc_test1 = []
    yc_test2 = []
    yc_test3 = []
    for i in range(len(x_test[:, 0])):
        if x_test[i, 1] == 3:
            x_test3 = np.append(x_test3, x_test[i, :].copy().reshape(1, xlength), axis=0)
            yd_test3 = np.append(yd_test3, yd_test[i].copy().reshape(1), axis=0)
            yc_test3 = np.append(yc_test3, yc_test[i].copy().reshape(1), axis=0)
        else:
            if x_test[i, 1] == 2:
                x_test2 = np.append(x_test2, x_test[i, :].copy().reshape(1, xlength), axis=0)
                yd_test2 = np.append(yd_test2, yd_test[i].copy().reshape(1), axis=0)
                yc_test2 = np.append(yc_test2, yc_test[i].copy().reshape(1), axis=0)
            else:
                x_test1 = np.append(x_test1, x_test[i, :].copy().reshape(1, xlength), axis=0)
                yd_test1 = np.append(yd_test1, yd_test[i].copy().reshape(1), axis=0)
                yc_test1 = np.append(yc_test1, yc_test[i].copy().reshape(1), axis=0)
    return x_test1, x_test2, x_test3, yd_test1, yd_test2, yd_test3, yc_test1, yc_test2, yc_test3

start = time.perf_counter()

# return posterior probability distribution of each seller's latent variable
def sellerlatent(x_train, yd_train, yc_train, Lset, pi, sigma):
    infolist = pd.DataFrame({'anon_slr_id': x_train.copy()[:, 0]})
    for ilatent in range(len(Lset)):
        x_trainaug = x_train.copy()[:, 2:len(x_train)]
        Latentvector1 = np.array(Lset[ilatent])
        x_trainaug[:, 3:6] = Latentvector1.copy()
        x_trainaug[:, 9:12][x_train[:, 1] > 1] = Latentvector1.copy()
        x_trainaug[:, 15:18][x_train[:, 1] > 2] = Latentvector1.copy()
        x_train_struc = x_trainaug.copy().reshape(len(x_trainaug), -1, 6)
        x_train_struc[np.isnan(x_train_struc)] = -1
        yd_result, yc_result = model.predict(x_train_struc, verbose=0)
        yc_result = yc_result.reshape(-1)
        Py = np.zeros(shape=len(yd_train))
        Py[:][yd_train[:] == 0] = yd_result.copy()[:, 0][yd_train[:] == 0]
        Py[:][yd_train[:] == 1] = yd_result.copy()[:, 1][yd_train[:] == 1]
        Py[:][yd_train[:] == 2] = yd_result.copy()[:, 2][yd_train[:] == 2]
        Py_aid_yc = np.exp(- np.square(yc_result - yc_train) / 2 / sigma / sigma)
        Py[:][yc_train[:] != -1] = np.multiply(Py[:][yc_train[:] != -1], Py_aid_yc[:][yc_train[:] != -1])
        infolist = pd.concat([infolist, pd.DataFrame(Py.copy())], axis=1)
    infolist_mul = infolist
    infolist_mul = infolist_mul.set_index('anon_slr_id')
    infolist_mul.columns = np.arange(len(Lset))
    infolist_mul.iloc[:, 0:len(Lset)] = infolist_mul.iloc[:, 0:len(Lset)].apply(lambda xx: xx.apply(Decimal))
    infolist_mul = infolist_mul.clip((pow(10, Decimal(-20))), 100)  # The purpose is to avoid having zero Py value after float calculation
    infocomb = infolist_mul.groupby('anon_slr_id').prod()
    depi = [Decimal(ini) for ini in pi]
    sellerlest = infocomb.mul(depi, axis=1)
    deno = [sum(sellerlest.iloc[inj, :]) for inj in range(len(sellerlest.iloc[:, 0]))]
    sellerlest_norm = sellerlest.div(deno, axis=0)
    sellerlest_norm.iloc[:, 0:len(Lset)] = sellerlest_norm.iloc[:, 0:len(Lset)].apply(lambda xxx: xxx.apply(float))
    return sellerlest_norm

def augmentxtrain_weighted(x_train, yd_train, yc_train, Lset, sellerlest_norm):
    interlframe = sellerlest_norm.copy()
    xtrain_aidlist = pd.DataFrame({'anon_slr_id': x_train.copy()[:, 0]})
    xtrain_aidlist = xtrain_aidlist.set_index('anon_slr_id')
    for intaid in range(len(Lset)):
        xtrain_aidlist[intaid] = np.nan
    combinelist = pd.concat([xtrain_aidlist, interlframe], axis=0)
    for intaid in range(len(Lset)):
        combinelist[intaid] = combinelist[intaid].fillna(combinelist.groupby(['anon_slr_id'])[intaid].bfill())
        combinelist[intaid] = combinelist[intaid].fillna(pi[intaid])
    weight_of_data = np.array(combinelist.iloc[0:len(xtrain_aidlist[0]), 0:len(Lset)]).reshape(len(xtrain_aidlist[0])*len(Lset))
    # create yd_train_expand and yc_train_expand
    yd_train_aid = yd_train.copy()
    yd_train_expand = np.repeat(yd_train_aid, len(Lset), axis=0)
    yc_train_aid = yc_train.copy()
    yc_train_expand = np.repeat(yc_train_aid, len(Lset), axis=0)
    # create x_train_withl_expand
    x_train_aid = x_train.copy()[:, 2:len(x_train)]
    x_train_aid = np.repeat(x_train_aid, len(Lset), axis=0)
    for intaid in range(len(Lset)):
        Latentvector2 = np.array(Lset[intaid])
        x_train_aid[intaid::len(Lset), 3:6] = Latentvector2.copy()
        if x_train_aid[intaid::len(Lset), 9:12][x_train[:, 1] > 1].shape[1] > 0:
            x_train_aid[intaid::len(Lset), 9:12][x_train[:, 1] > 1] = Latentvector2.copy()
        if x_train_aid[intaid::len(Lset), 15:18][x_train[:, 1] > 2].shape[1] > 0:
            x_train_aid[intaid::len(Lset), 15:18][x_train[:, 1] > 2] = Latentvector2.copy()
    x_train_withl_expand = x_train_aid.copy().reshape(len(x_train_aid), -1, 6)
    x_train_withl_expand[np.isnan(x_train_withl_expand)] = -1
    return x_train_withl_expand, yd_train_expand, yc_train_expand, weight_of_data

def augmentxtrain_weighted_sample_loop(x_train, yd_train, yc_train, Lset, sellerlest_norm, sample_size, iterationind, loopnumber):
    interlframe = sellerlest_norm.copy()
    xtrain_aidlist = pd.DataFrame({'anon_slr_id': x_train.copy()[:, 0]})
    xtrain_aidlist = xtrain_aidlist.set_index('anon_slr_id')
    for intaid in range(len(Lset)):
        xtrain_aidlist[intaid] = np.nan
    combinelist = pd.concat([xtrain_aidlist, interlframe], axis=0)
    for intaid in range(len(Lset)):
        combinelist[intaid] = combinelist[intaid].fillna(combinelist.groupby(['anon_slr_id'])[intaid].bfill())
        combinelist[intaid] = combinelist[intaid].fillna(pi[intaid])
    weight_of_data_exact = np.array(combinelist.iloc[0:len(xtrain_aidlist[0]), 0:len(Lset)])
    weight_of_data = np.empty((len(weight_of_data_exact[:, 0]), len(Lset[:, 0])))
    for intsample in range(len(weight_of_data_exact[:, 0])):
        sample_vector = np.random.choice(len(Lset[:, 0]), size=sample_size, p=weight_of_data_exact[intsample, :])
        weight_of_data[intsample, :] = 1 / sample_size * np.bincount(sample_vector, minlength=len(Lset[:, 0]))
    weight_of_data = weight_of_data.reshape(len(xtrain_aidlist[0])*len(Lset))
    # set weights of data points (with all latent vectors) that are not selected in this round to zeros
    keepindex = iterationind % loopnumber
    fillvector = np.zeros(loopnumber)
    fillvector[keepindex] = 1
    fillvector = np.tile(fillvector, np.int64(np.ceil(len(xtrain_aidlist[0]) / loopnumber)))
    fillvector = np.repeat(fillvector, len(Lset), axis=0)
    weight_of_data = weight_of_data * fillvector[0:len(weight_of_data)]
    # create yd_train_expand and yc_train_expand
    yd_train_aid = yd_train.copy()
    yd_train_expand = np.repeat(yd_train_aid, len(Lset), axis=0)
    yc_train_aid = yc_train.copy()
    yc_train_expand = np.repeat(yc_train_aid, len(Lset), axis=0)
    # create x_train_withl_expand
    x_train_aid = x_train.copy()[:, 2:len(x_train)]
    x_train_aid = np.repeat(x_train_aid, len(Lset), axis=0)
    for intaid in range(len(Lset)):
        Latentvector2 = np.array(Lset[intaid])
        x_train_aid[intaid::len(Lset), 3:6] = Latentvector2.copy()
        if x_train_aid[intaid::len(Lset), 9:12][x_train[:, 1] > 1].shape[1] > 0:
            x_train_aid[intaid::len(Lset), 9:12][x_train[:, 1] > 1] = Latentvector2.copy()
        if x_train_aid[intaid::len(Lset), 15:18][x_train[:, 1] > 2].shape[1] > 0:
            x_train_aid[intaid::len(Lset), 15:18][x_train[:, 1] > 2] = Latentvector2.copy()
    x_train_withl_expand = x_train_aid.copy().reshape(len(x_train_aid), -1, 6)
    x_train_withl_expand[np.isnan(x_train_withl_expand)] = -1
    return x_train_withl_expand, yd_train_expand, yc_train_expand, weight_of_data

def f_test_accuracy(x_test, yd_test, yc_test, Lset, sellerlest_norm):
    x_test_aid = x_test.copy()[:, 2:len(x_test)]
    x_test_aid = np.repeat(x_test_aid, len(Lset), axis=0)
    for intaid in range(len(Lset)):
        Latentvector3 = np.array(Lset[intaid])
        x_test_aid[intaid::len(Lset), 3:6] = Latentvector3.copy()
        x_test_aid[intaid::len(Lset), 9:12][x_test[:, 1] > 1] = Latentvector3.copy()
        x_test_aid[intaid::len(Lset), 15:18][x_test[:, 1] > 2] = Latentvector3.copy()
    x_test_withl_expand = x_test_aid.copy().reshape(len(x_test_aid), -1, 6)
    x_test_withl_expand[np.isnan(x_test_withl_expand)] = -1
    yd_pred_result, yc_pred_result = model.predict(x_test_withl_expand, verbose=0)
    yc_pred_result = yc_pred_result.reshape(-1)
    # Calculate the weight of prediction under each latent value
    interlframe_test = sellerlest_norm.copy()
    xtest_aidlist = pd.DataFrame({'anon_slr_id': x_test.copy()[:, 0]})
    xtest_aidlist = xtest_aidlist.set_index('anon_slr_id')
    for intaid_test in range(len(Lset)):
        xtest_aidlist[intaid_test] = np.nan
    combinelist_test = pd.concat([xtest_aidlist, interlframe_test], axis=0)
    for intaid_test in range(len(Lset)):
        combinelist_test[intaid_test] = combinelist_test[intaid_test].fillna(combinelist_test.groupby(['anon_slr_id'])[intaid_test].bfill())
        combinelist_test[intaid_test] = combinelist_test[intaid_test].fillna(pi[intaid_test])
    weight_of_test = np.array(combinelist_test.iloc[0:len(xtest_aidlist[0]), 0:len(Lset)]).reshape(len(xtest_aidlist[0])*len(Lset))
    # Calculate yd_pred and the accuracy, F1score
    yd_pred_result_frame = pd.DataFrame(yd_pred_result)
    yd_pred_matrix = yd_pred_result_frame.mul(weight_of_test, axis=0)
    yd_pred_combine = yd_pred_matrix.groupby(yd_pred_matrix.index // len(Lset)).sum()
    yd_pred = np.array(yd_pred_combine).reshape(-1, 3)
    yd_pred_arg = yd_pred.argmax(axis=1)
    compare = np.equal(yd_test, yd_pred_arg)
    test_accuracy = sum(compare)/len(compare)
    F1score_weighted = f1_score(yd_test, yd_pred_arg, average='weighted')
    # Calculate yc_pred and the mse
    yc_pred_result_frame = pd.DataFrame(yc_pred_result)
    yc_pred_matrix = yc_pred_result_frame.mul(weight_of_test, axis=0)
    yc_pred_combine = yc_pred_matrix.groupby(yc_pred_matrix.index // len(Lset)).sum()
    yc_pred = np.array(yc_pred_combine).reshape(-1)
    mse_value = calculatemse(yc_test, yc_pred)
    return test_accuracy, F1score_weighted, mse_value

def Correlation_LandP(x_test, Lset, numberofsamples, numberwithinsample):
    x_test_len = len(x_test[:, 0])
    x_test_index_list = list(np.arange(x_test_len))
    x_sample = random.sample(x_test_index_list, numberofsamples)
    R0 = np.zeros(numberofsamples)
    R1 = np.zeros(numberofsamples)
    R2 = np.zeros(numberofsamples)
    R3 = np.zeros(numberofsamples)
    R4 = np.zeros(numberofsamples)
    R5 = np.zeros(numberofsamples)
    R6 = np.zeros(numberofsamples)
    R7 = np.zeros(numberofsamples)
    R8 = np.zeros(numberofsamples)
    R00 = np.zeros(numberofsamples)
    R01 = np.zeros(numberofsamples)
    R02 = np.zeros(numberofsamples)
    for i in range(numberofsamples):
        x_test_chosen = x_test[x_sample[i], 2:len(x_test)].reshape(1, -1)
        x_test_chosen_expand = np.repeat(x_test_chosen, numberwithinsample, axis=0)
        Latentvector_cr = np.empty((numberwithinsample, 3))
        Latentvector_cr[:, 0] = np.array(truncnorm.rvs(0, 0.01, size=numberwithinsample))
        Latentvector_cr[:, 1] = np.array(truncnorm.rvs(0, 0.01, size=numberwithinsample))
        Latentvector_cr[:, 2] = np.array(truncnorm.rvs(0, 0.01, size=numberwithinsample))
        x_test_chosen_expand[:, 3:6] = Latentvector_cr.copy()
        x_test_chosen_expand[:, 9:12][x_test[x_sample[i], 1] > 1] = Latentvector_cr.copy()
        x_test_chosen_expand[:, 15:18][x_test[x_sample[i], 1] > 2] = Latentvector_cr.copy()
        x_test_chosen_expand_withl = x_test_chosen_expand.copy().reshape(len(x_test_chosen_expand), -1, 6)
        x_test_chosen_expand_withl[np.isnan(x_test_chosen_expand_withl)] = -1
        yd_pred_result, yc_pred_result = model.predict(x_test_chosen_expand_withl, verbose=0)
        yc_pred_result = yc_pred_result.reshape(-1)
        d_0 = {'L1': Latentvector_cr.copy()[:, 0], 'L2': Latentvector_cr.copy()[:, 1], 'L3': Latentvector_cr.copy()[:, 2], 'Accept': yd_pred_result.copy()[:, 0]}
        combine_LandP_0 = pd.DataFrame(data=d_0)
        d_1 = {'L1': Latentvector_cr.copy()[:, 0], 'L2': Latentvector_cr.copy()[:, 1], 'L3': Latentvector_cr.copy()[:, 2], 'Reject': yd_pred_result.copy()[:, 1]}
        combine_LandP_1 = pd.DataFrame(data=d_1)
        d_2 = {'L1': Latentvector_cr.copy()[:, 0], 'L2': Latentvector_cr.copy()[:, 1], 'L3': Latentvector_cr.copy()[:, 2], 'Counter': yd_pred_result.copy()[:, 2]}
        combine_LandP_2 = pd.DataFrame(data=d_2)
        R0[i] = pg.partial_corr(data=combine_LandP_0, x='L1', y='Accept', covar=['L2','L3']).iloc[0, 1]
        R1[i] = pg.partial_corr(data=combine_LandP_1, x='L1', y='Reject', covar=['L2','L3']).iloc[0, 1]
        R2[i] = pg.partial_corr(data=combine_LandP_2, x='L1', y='Counter', covar=['L2','L3']).iloc[0, 1]
        R3[i] = pg.partial_corr(data=combine_LandP_0, x='L2', y='Accept', covar=['L1','L3']).iloc[0, 1]
        R4[i] = pg.partial_corr(data=combine_LandP_1, x='L2', y='Reject', covar=['L1','L3']).iloc[0, 1]
        R5[i] = pg.partial_corr(data=combine_LandP_2, x='L2', y='Counter', covar=['L1','L3']).iloc[0, 1]
        R6[i] = pg.partial_corr(data=combine_LandP_0, x='L3', y='Accept', covar=['L1','L2']).iloc[0, 1]
        R7[i] = pg.partial_corr(data=combine_LandP_1, x='L3', y='Reject', covar=['L1','L2']).iloc[0, 1]
        R8[i] = pg.partial_corr(data=combine_LandP_2, x='L3', y='Counter', covar=['L1','L2']).iloc[0, 1]
        d_yc = {'L1': Latentvector_cr.copy()[:, 0], 'L2': Latentvector_cr.copy()[:, 1], 'L3': Latentvector_cr.copy()[:, 2], 'counterprice': yc_pred_result.copy()}
        combine_LandP_yc = pd.DataFrame(data=d_yc)
        R00[i] = pg.partial_corr(data=combine_LandP_yc, x='L1', y='counterprice', covar=['L2','L3']).iloc[0, 1]
        R01[i] = pg.partial_corr(data=combine_LandP_yc, x='L2', y='counterprice', covar=['L1','L3']).iloc[0, 1]
        R02[i] = pg.partial_corr(data=combine_LandP_yc, x='L3', y='counterprice', covar=['L1','L2']).iloc[0, 1]
    # print(R0, R1, R2, R3, R4, R5, R6, R7, R8)
    print(np.mean(R0), np.mean(R1), np.mean(R2), np.mean(R3), np.mean(R4), np.mean(R5), np.mean(R6), np.mean(R7), np.mean(R8))
    print(np.std(R0), np.std(R1), np.std(R2), np.std(R3), np.std(R4), np.std(R5), np.std(R6), np.std(R7), np.std(R8))
    print(np.var(R0), np.var(R1), np.var(R2), np.var(R3), np.var(R4), np.var(R5), np.var(R6), np.var(R7), np.var(R8))
    print(np.mean(R00), np.mean(R01), np.mean(R02))
    print(np.std(R00), np.std(R01), np.std(R02))
    print(np.var(R00), np.var(R01), np.var(R02))

# main program

df = pd.read_csv('bargain_dataset.csv')

df = df.drop(['anon_item_id', 'anon_thread_id', 'fdbk_score_src', 'fdbk_pstv_src', 'slr_hist', 'byr_hist', 'to_lst_cnt', 'bo_lst_cnt', 'view_item_count', 'bin_rev'], axis=1)
df.loc[df['ys1'] == 1, 'ys'] = 1
df.loc[df['ys2'] == 1, 'ys'] = 2
df.loc[df['ys3'] == 1, 'ys'] = 3
df = df.drop(['ys1', 'ys2', 'ys3'], axis=1)

df.insert(5, 'l_round11', np.nan)
df.insert(6, 'l_round12', np.nan)
df.insert(7, 'l_round13', np.nan)
df.insert(11, 'l_round21', np.nan)
df.insert(12, 'l_round22', np.nan)
df.insert(13, 'l_round23', np.nan)
df.insert(17, 'l_round31', np.nan)
df.insert(18, 'l_round32', np.nan)
df.insert(19, 'l_round33', np.nan)

xpre = df.iloc[:, :-2].values
yc = df.iloc[:, -2].values
yd = df.iloc[:, -1].values

# convert yc into a value between 0 and 1, based on the latest two asking prices
for int in range(len(xpre[:, 0])):
    if yd[int] == 3:
        brecent, srecent = findrecentask(xpre[int])
        if brecent != srecent:
            yc_ratio = yc[int] / xpre[int, 2]  # xpre[int, 2] is s0
        else:
            yc_ratio = -100
        yc[int] = yc_ratio

le = preprocessing.LabelEncoder()
le.fit([1, 2, 3])
yd = le.transform(yd)

x_train, x_test, yc_train, yc_test, yd_train, yd_test = train_test_split(xpre, yc, yd, test_size=0.2, random_state=35)
x_train, x_val, yc_train, yc_val, yd_train, yd_val = train_test_split(x_train, yc_train, yd_train, test_size=0.25, random_state=35)

# normalization
sc = MinMaxScaler(feature_range=(0, 1))
x_train[:, 2:len(x_train)] = sc.fit_transform(x_train[:, 2:len(x_train)].reshape(-1, x_train[:, 2:len(x_train)].shape[-1])).reshape(x_train[:, 2:len(x_train)].shape)
x_test[:, 2:len(x_test)] = sc.transform(x_test[:, 2:len(x_test)].reshape(-1, x_test[:, 2:len(x_test)].shape[-1])).reshape(x_test[:, 2:len(x_test)].shape)
x_val[:, 2:len(x_val)] = sc.transform(x_val[:, 2:len(x_val)].reshape(-1, x_val[:, 2:len(x_val)].shape[-1])).reshape(x_val[:, 2:len(x_val)].shape)

yc_train[np.isnan(yc_train)] = -1
yc_test[np.isnan(yc_test)] = -1
yc_val[np.isnan(yc_val)] = -1

# classify x_test and yd_test, yc_test based on the number of rounds
x_test1, x_test2, x_test3, yd_test1, yd_test2, yd_test3, yc_test1, yc_test2, yc_test3 = classifytestdata(x_test, yd_test, yc_test)

numberofiterations = 50
number_epochs = 200
sigma = (1/200) ** 0.5  # set hyper-parameter sigma

sample_size = 4
loopnumber = 3

Lset = np.empty((64, 3))
pi = np.empty(64)

for i in range(64):
    pi[i] = 1/64
    Lset[i, 2] = 0.01 / 3 * (i // 16)
    Lset[i, 1] = 0.01 / 3 * ((i % 16) // 4)
    Lset[i, 0] = 0.01 / 3 * ((i % 16) % 4)

print(Lset)

# Three Layer Network
inputs = tf.keras.layers.Input(shape=(3, 6))
masking = tf.keras.layers.Masking(mask_value=-1)(inputs)
hidden1 = tf.keras.layers.LSTM(8, return_sequences=True)(masking)
hidden2 = tf.keras.layers.LSTM(9, return_sequences=True)(hidden1)
hidden3 = tf.keras.layers.LSTM(8, return_sequences=False)(hidden2)
out_class = Dense(3, activation='softmax', name='decision')(hidden3)
hidden4 = Dense(7, activation='sigmoid')(hidden3)
out_reg = Dense(1, activation='sigmoid', name='counterprice')(hidden4)

model = tf.keras.models.Model(inputs=inputs, outputs=[out_class, out_reg])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), masked_mse],
              loss_weights={"decision": 1, "counterprice": 1/2/(sigma * sigma)},
              weighted_metrics={'decision': 'sparse_categorical_accuracy'})

val_accuracyvector_woft = np.zeros(numberofiterations)
val_msevector_woft = np.zeros(numberofiterations)
test_accuracyvector_woft = np.zeros(numberofiterations)
test_accuracyvector1_woft = np.zeros(numberofiterations)
test_accuracyvector2_woft = np.zeros(numberofiterations)
test_accuracyvector3_woft = np.zeros(numberofiterations)
F1score_weighted_vector_woft = np.zeros(numberofiterations)
test_msevector_woft = np.zeros(numberofiterations)
test_msevector1_woft = np.zeros(numberofiterations)
test_msevector2_woft = np.zeros(numberofiterations)
test_msevector3_woft = np.zeros(numberofiterations)

elapsedvector = np.zeros(numberofiterations)
testtimevector = np.zeros(numberofiterations)

for iterationind in range(numberofiterations):

    sellerlest_norm = sellerlatent(x_train.copy(), yd_train.copy(), yc_train.copy(), Lset, pi.copy(), sigma)
    sellerlest_norm.columns = np.arange(len(Lset))

    print(iterationind, sellerlest_norm.iloc[0:99, 0:len(Lset)])

    Phi = sellerlest_norm.sum(axis=0)

    pi = Phi/sum(Phi)
    pi = np.array(pi)

    # Training the model and update the weights
    x_train_withl_expand, yd_train_expand, yc_train_expand, weight_of_data = augmentxtrain_weighted_sample_loop(x_train.copy(), yd_train.copy(), yc_train.copy(), Lset, sellerlest_norm.copy(), sample_size, iterationind, loopnumber)
    x_train_withl_expand, yd_train_expand, yc_train_expand, weight_of_data = shuffle(x_train_withl_expand, yd_train_expand, yc_train_expand, weight_of_data)

    # remove training data if the reassigned weights are zeros
    print(len(weight_of_data))
    threshold_sampleweight = 1e-5
    weight_of_data_store = weight_of_data.copy()
    x_train_withl_expand = np.delete(x_train_withl_expand.reshape(-1, 18), np.where(weight_of_data_store < threshold_sampleweight), axis=0).reshape(-1, 3, 6)
    yd_train_expand = np.delete(yd_train_expand, np.where(weight_of_data_store < threshold_sampleweight))
    yc_train_expand = np.delete(yc_train_expand, np.where(weight_of_data_store < threshold_sampleweight))
    weight_of_data = np.delete(weight_of_data, np.where(weight_of_data_store < threshold_sampleweight))
    print(len(weight_of_data))

    # x_val_withl_expand, yd_val_expand, yc_val_expand, val_weight_of_data = augmentxtrain_weighted(x_val.copy(), yd_val.copy(), yc_val.copy(), Lset, sellerlest_norm.copy())
    x_val_withl_expand, yd_val_expand, yc_val_expand, val_weight_of_data = augmentxtrain_weighted_sample_loop(x_val.copy(), yd_val.copy(), yc_val.copy(), Lset, sellerlest_norm.copy(), 5, iterationind, 1)
    x_val_withl_expand, yd_val_expand, yc_val_expand, val_weight_of_data = shuffle(x_val_withl_expand, yd_val_expand, yc_val_expand, val_weight_of_data)

    # remove validation data if the reassigned weights are zeros
    val_weight_of_data_store = val_weight_of_data.copy()
    x_val_withl_expand = np.delete(x_val_withl_expand.reshape(-1, 18), np.where(val_weight_of_data_store < threshold_sampleweight), axis=0).reshape(-1, 3, 6)
    yd_val_expand = np.delete(yd_val_expand, np.where(val_weight_of_data_store < threshold_sampleweight))
    yc_val_expand = np.delete(yc_val_expand, np.where(val_weight_of_data_store < threshold_sampleweight))
    val_weight_of_data = np.delete(val_weight_of_data, np.where(val_weight_of_data_store < threshold_sampleweight))

    cp_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(x_train_withl_expand, [yd_train_expand, yc_train_expand], sample_weight=weight_of_data, validation_data=(x_val_withl_expand, [yd_val_expand, yc_val_expand], val_weight_of_data), batch_size=128, validation_batch_size=50000000, epochs=number_epochs, callbacks=[cp_callback])

    model.summary()

    test_begin_time = time.perf_counter()

    val_accuracy_woft, val_F1score_weighted_woft, val_mse_woft = f_test_accuracy(x_val.copy(), yd_val.copy(), yc_val.copy(), Lset, sellerlest_norm)
    val_accuracyvector_woft[iterationind] = val_accuracy_woft
    val_msevector_woft[iterationind] = val_mse_woft

    test_accuracy_woft, test_F1score_weighted_woft, test_mse_woft = f_test_accuracy(x_test.copy(), yd_test.copy(), yc_test.copy(), Lset, sellerlest_norm)
    test_accuracyvector_woft[iterationind] = test_accuracy_woft
    F1score_weighted_vector_woft[iterationind] = test_F1score_weighted_woft
    test_msevector_woft[iterationind] = test_mse_woft

    test_accuracy1_woft, test_F1score_weighted_1_woft, test_mse1_woft = f_test_accuracy(x_test1.copy(), yd_test1.copy(), yc_test1.copy(), Lset, sellerlest_norm.copy())
    test_accuracyvector1_woft[iterationind] = test_accuracy1_woft
    test_msevector1_woft[iterationind] = test_mse1_woft

    test_accuracy2_woft, test_F1score_weighted_2_woft, test_mse2_woft = f_test_accuracy(x_test2.copy(), yd_test2.copy(), yc_test2.copy(), Lset, sellerlest_norm.copy())
    test_accuracyvector2_woft[iterationind] = test_accuracy2_woft
    test_msevector2_woft[iterationind] = test_mse2_woft

    test_accuracy3_woft, test_F1score_weighted_3_woft, test_mse3_woft = f_test_accuracy(x_test3.copy(), yd_test3.copy(), yc_test3.copy(), Lset, sellerlest_norm.copy())
    test_accuracyvector3_woft[iterationind] = test_accuracy3_woft
    test_msevector3_woft[iterationind] = test_mse3_woft

    test_end_time = time.perf_counter()

    testtimevector[iterationind] = test_end_time - test_begin_time

    print(val_accuracyvector_woft, val_msevector_woft)
    print(test_accuracyvector_woft, test_accuracyvector1_woft, test_accuracyvector2_woft, test_accuracyvector3_woft)
    print(test_msevector_woft, test_msevector1_woft, test_msevector2_woft, test_msevector3_woft)
    print(F1score_weighted_vector_woft)

    elapsed = time.perf_counter() - start - sum(testtimevector[0: (iterationind + 1)])
    print('Elapsed %.3f seconds.' % elapsed)
    elapsedvector[iterationind] = elapsed
    print(elapsedvector)

    # Calculate Distribution of Correlation Coefficient Between Latent Variables and Strategy Probability
    # numberofsamples = 800
    # numberwithinsample = 3000
    # if (iterationind == 0 or iterationind == 4 or iterationind == 9 or iterationind == 14 or iterationind == 19 or iterationind == 24 or iterationind == 29 or iterationind == 34 or iterationind == 39 or iterationind == 49):
        # Correlation_LandP(x_test.copy(), Lset, numberofsamples, numberwithinsample)
