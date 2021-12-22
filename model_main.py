# from model_factory.hdbscan import hdbscan_model
from model_factory.rrcf import rrcf_model
from model_factory.IForest import IForest_model
from model_factory.HBOS import HBOS_model
# from model_factory.KNN import KNN_model
from model_factory.CBLOF import CBLOF_model
from model_factory.COPOD import COPOD_model
# from model_factory.LOF import LOF_model
# from misc.misc import X_modeller,autodetector,X_predictor,autodetector_predict,setup_logger
from misc.misc import autodetector
from .optimization_main import run_optimization
from misc.hp_config  import deflt_hps
from misc.hp_config  import deflt_hps_rng
from joblib import Parallel, delayed, wrap_non_picklable_objects

import logging
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pyod.models.combination import aom, moa, average, maximization


class run_algorithm:

    """
    run_algorithm utilize model factory from auto_AD package
    algorithms available from model factory: IForest_model, CBLOF_model,
    KNN_model,HBOS_model,hdbscan_model,LOF_model,COPOD_model

    Parameters:
        X (array): is the data to train the algorithm
        algorithm_name(str): algorithm to train the data
        hps (dictionary): algorithm hyperparamters
        remove_zeros(True/False) : remove the cardial rows and assign zero score
        normalization(True/False):Normalizes the input data before modeling

    Returns:
        anomaly scores and model objects
    """
    def __init__(self,algorithms,remove_zeros=True,
                 normalization=True,ensemble="Average",n_jobs=-1,hps=None,i_logger=None,best_hps=False):
        self.algorithms = algorithms
        self.remove_zeros =  remove_zeros
        self.normalization = normalization
        self.ensemble = ensemble
        self.n_jobs = n_jobs
        self.score_df = pd.DataFrame()
        self.clf_dict = {}
        self.scaling_dict = {"scaler_X":{},"scaler_anomalies":{}}
        self.hps = hps
        self.i_logger = i_logger
        self.best_hps = best_hps

    def run_ensemble(self,X=None):
        try:
            if self.best_hps==True:
                hps_default = deflt_hps()
                if self.hps ==None:
                    self.hps = {"algorithms":{}}
                for algorithm in self.algorithms:
                    storage_name = "test_db"
                    study_name = "test_storage_"+algorithm
                    try:
                        run = run_optimization(X,storage_name,
                                             study_name,
                                             direction = "maximize" ,
                                             n_trials = 64,
                                             trial_parallel = True,
                                             algorithm = algorithm)
                        b_hps = str(run.best_trial.params)
                        self.i_logger.info(f"Optimization succesfull and best hps for {algorithm} {b_hps}")
                    except:
                        b_hps = hps_default["algorithms"][algorithm]
                    self.hps["algorithms"][algorithm] = b_hps
            allscores = autodetector(X,self.algorithms,self.remove_zeros,None,None,
                                     self.normalization,self.n_jobs,self.hps,self.i_logger)
            # print(allscores)
            self.failed_algorithms = []
            self.success_algorithms = []
            scale_check = True
            for index in range(0,len(self.algorithms)):
                if allscores[index]["run"] == "success":
                    algorithm = allscores[index]["algorithm"]
                    self.success_algorithms.append(algorithm)
                    self.score_df[algorithm] = allscores[index]["anomaly_score_scaled"]*100
                    self.score_df[algorithm] = self.score_df[algorithm].round(2)
                    self.clf_dict[self.algorithms[index]] = allscores[index]["clf_fit"]
                    if scale_check == True:
                        self.scaling_dict["scaler_X"] = allscores[index]["scaler_X"]
                        scale_check = False
                    self.scaling_dict["scaler_anomalies"][self.algorithms[index]] = allscores[index]["scaler_anmls"]
                else:
                    listofzeros = ["NA"] * X.shape[0]
                    self.failed_algorithms.append(self.algorithms[index])
                    algorithm = self.algorithms[index]
                    self.score_df[algorithm] = listofzeros
            score_ensemble = eval(self.ensemble)(self.score_df[self.success_algorithms]).round(2)
            self.score_df.columns = ["Anomaly Score "+x.replace("_model","") for x in self.score_df.columns]
            self.score_df["Anomaly Score Ensemble"] =  score_ensemble
        except:
            print("Failure")
            self.i_logger.exception("Exception occured")
        return self
    def inference(self,y):
        score_predict = autodetector_predict(y,self.clf_dict,self.scaling_dict,
                                             self.success_algorithms,self.normalization,self.remove_zeros,self.i_logger)
        inference = pd.DataFrame()
        for index in range(0,len(self.success_algorithms)):
            algorithm = score_predict[index]["algorithm"]
    #             algorithm = self.success_algorithms[index]
            inference[algorithm] =score_predict[index]["anomaly_score_scaled"].reshape(-1)*100
            inference[algorithm] = inference[algorithm].round(2)
        score_ensemble = eval(self.ensemble)(inference).round(2)
        inference.columns = ["Anomaly Score "+x.replace("_model","") for x in inference.columns]
        inference["Anomaly Score Ensemble"] = score_ensemble
        return inference
