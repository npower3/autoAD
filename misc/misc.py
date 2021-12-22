# from auto_AD.model_factory.hdbscan import hdbscan_model
from auto_AD.model_factory.rrcf import rrcf_model
from auto_AD.model_factory.IForest import IForest_model
from auto_AD.model_factory.HBOS import HBOS_model
# from auto_AD.model_factory.KNN import KNN_model
from auto_AD.model_factory.CBLOF import CBLOF_model
from auto_AD.model_factory.COPOD import COPOD_model
# from auto_AD.model_factory.LOF import LOF_model

from auto_AD.misc.hp_config import deflt_hps
from auto_AD.misc.hp_config import deflt_hps_rng

import pickle
from sklearn.model_selection import ShuffleSplit
import numpy as np
from numpy import isnan
import pandas  as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import logging
from pyod.models.combination import aom, moa, average, maximization
import sys
from joblib import Parallel, delayed

import logging


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    #     handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    #     console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class static_variable():
    def _init_(self, config):
        self.algorithms = list(config["algorithms"])
        self.hyper_params = config["algorithms"]
        self.data_location = config['data_locations']["location"]
        self.model_chunks = config['data_locations']["model_chunks"]
        self.cv = ShuffleSplit(n_splits=config["tune_config"]["kfold"]["n_splits"],
                               test_size=config["tune_config"]["kfold"]["test_size"], random_state=42)
        self.n_sim = config["mv_config"]["n_sim"]
        self.algorithms_run = config["algorithms_run"]
        self.curve = config["curve"]
        self.hps_loc = config["optimizer_config"]["hps_loc"]
        self.optimizer_config = config["optimizer_config"]["n_trials"]
        self.direction = config["optimizer_config"]["direction"]
        self.level = config["level"]
        self.optimzation_condition = config["optimizer_config"]["optimization_run"]


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def X_scale(X, scaler):
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, scaler


def X_modeller(X, algorithm, remove_zeros, clf_fit,
               scaling_dict, normalization=True, hps=None, i_logger=None):
    #     i_logger.info(f"Run Started for {algorithm}")
    try:
        hps_default = deflt_hps()
        hps_default = eval(hps_default["algorithms"][algorithm])
        if hps != None:
            hps_best = eval(hps["algorithms"][algorithm])
            if len(hps_best) > 1:
                hps_default.update(hps_best)
                print(hps_default)

        clf = eval(algorithm)(hps_default)
        X = X.reshape(len(X), -1)
        if remove_zeros == True:
            nnzro_ind = np.where(np.abs(X).sum(axis=1) != 0)[0]
            zro_ind = np.where(np.abs(X).sum(axis=1) == 0)[0]
            if normalization == True:
                scaler_X = StandardScaler()
                scaler_X.fit(X[nnzro_ind])
                X_scaled = scaler_X.transform(X[nnzro_ind])
            else:
                scaler_X = None
                X_scaled = X[nnzro_ind]
            clf_fit = clf.fit(X_scaled)
            anomaly_score = clf_fit.decision_function(X_scaled)
            anomaly_score = clf_fit.decision_function(X_scaled)
            anomaly_score[isnan(anomaly_score)] = 0
            scaler_anmls = MinMaxScaler()
            scaler_anmls.fit(anomaly_score.reshape(-1, 1))
            anomaly_score_scaled = scaler_anmls.transform(anomaly_score.reshape(-1, 1))
            mstr_X = pd.DataFrame(X)
            anmy_scr = pd.DataFrame(anomaly_score_scaled)
            anmy_scr.index = nnzro_ind
            mstr_X = pd.merge(mstr_X, anmy_scr, left_index=True, right_index=True, how="left")
            anomaly_score_scaled = mstr_X.fillna(0).iloc[:, -1].values

        else:
            if normalization == True:
                scaler_X = StandardScaler()
                scaler_X.fit(X)
                X_scaled = scaler_X.transform(X)
            else:
                scaler_X = None
                X_scaled = X
            clf_fit = clf.fit(X_scaled)
            anomaly_score = clf_fit.decision_function(X_scaled)
            anomaly_score[isnan(anomaly_score)] = 0
            scaler_anmls = MinMaxScaler()
            scaler_anmls.fit(anomaly_score.reshape(-1, 1))
            anomaly_score_scaled = scaler_anmls.transform(anomaly_score.reshape(-1, 1)).reshape(-1, )
        model_dict = {"clf_fit": clf_fit, "scaler_X": scaler_X, "scaler_anmls": scaler_anmls,
                      "anomaly_score_scaled": anomaly_score_scaled, "algorithm": algorithm, "run": "success"}
        run = "success"
    except:
        run = "failed"
        i_logger.exception("Exception occured")
        model_dict = {"clf_fit": None, "scaler_X": None, "scaler_anmls": None, "anomaly_score_scaled": None,
                      "algorithm": algorithm, "run": "Fail"}
        i_logger.info(f"Algorithm run {run} for {algorithm}")
    return model_dict


def autodetector(X, algorithms, remove_zeros, clf_dict, scaler_dict, normalization, n_jobs, hps, i_logger):
    i_logger.info(f"Parallelization started for {algorithms}")
    allscores = Parallel(n_jobs=n_jobs)(delayed(
        X_modeller)(X, algorithm, remove_zeros, None, None, normalization, hps, i_logger)
                                        for algorithm in algorithms)
    # print(allscores)
    return allscores


def X_predictor(y, clf_fit, scaler_X, scaler_anomaly, remove_zeros, normalization, algorithm, i_logger):
    nnzro_ind = np.where(np.abs(y).sum(axis=1) != 0)[0]
    zro_ind = np.where(np.abs(y).sum(axis=1) == 0)[0]
    if remove_zeros == False:
        nnzro_ind = np.concatenate((nnzro_ind, zro_ind))
    if normalization == True:
        y_scaled = scaler_X.transform(y[nnzro_ind])
    else:
        y_scaled = y
    anomaly_score = clf_fit.decision_function(y_scaled)
    anomaly_score[isnan(anomaly_score)] = 0
    anomaly_score_scaled = scaler_anomaly.transform(anomaly_score.reshape(-1, 1))
    mstr_X = pd.DataFrame(y)
    anmy_scr = pd.DataFrame(anomaly_score_scaled)
    anmy_scr.index = nnzro_ind
    mstr_X = pd.merge(mstr_X, anmy_scr, left_index=True, right_index=True, how="left")
    anomaly_score_scaled = mstr_X.fillna(0).iloc[:, -1].values
    i_logger.info(f"Algorithm inference success {algorithm}")
    return {"anomaly_score_scaled": anomaly_score_scaled, "algorithm": algorithm}


def autodetector_predict(y, clf_dict, scaling_dict,
                         algorithms, normalization, remove_zeros, i_logger):
    allscores = Parallel(n_jobs=-1)(delayed(X_predictor)(y, clf_dict[algorithm], scaling_dict["scaler_X"],
                                                         scaling_dict["scaler_anomalies"][algorithm],
                                                         remove_zeros, normalization, algorithm, i_logger)
                                    for algorithm in algorithms)
    return allscores
