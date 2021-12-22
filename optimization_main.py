import pandas as pd
import numpy as np
import optuna
import warnings
from .misc.hp_config import deflt_hps
from .misc.hp_config import deflt_hps_rng
# from .model_main import run_algorithm
from importlib import reload
from .ad_metrics.em import em_metric
from joblib import Parallel, delayed
# from .model_factory.hdbscan import hdbscan_model
from .model_factory.rrcf import rrcf_model
from .model_factory.IForest import IForest_model
from .model_factory.HBOS import HBOS_model
# from .model_factory.KNN import KNN_model
from .model_factory.CBLOF import CBLOF_model
from .model_factory.COPOD import COPOD_model
# from .model_factory.LOF import LOF_model
from sklearn.model_selection import ShuffleSplit
import yaml

# with open(r'config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)


config = {'name': 'Config',
          'optimizer_config': {'frame_work': {'curve': 'em_metric',
                                              'dim_hps': True,
                                              'direction': 'maximize',
                                              'n_trials': 80},
                               'tune_config': {'kfold': {'n_sim': 10000,
                                                         'n_splits': 10,
                                                         'test_size': 0.2}}},
          'pipeline': {'algorithms': ['IForest_model',
                                      'KNN_model',
                                      'LOF_model',
                                      'hdbscan_model',
                                      'CBLOF_model'],
                       'algorithms_default': ['IForest_model'],
                       'algorithms_run': 'None'}}


class variable_obj:
    def _init_(self, hps, algorithm, curve, n_trials, direction):
        self.hps = hps
        self.curve = curve
        self.algorithms_run = algorithm
        self.n_trials = n_trials
        self.direction = direction


def objective(trial, X, static_variable_obj):
    hps = static_variable_obj.hps
    static_variable_obj.clf = eval(static_variable_obj.algorithms_run)(eval(hps))
    eval_metric_object = eval(static_variable_obj.curve)(X, static_variable_obj)
    eval_metric = eval_metric_object.metric_trial()
    return eval_metric


def optimize(n_trials, study_name, X, storage_name, static_variable_obj):
    #############################################################################
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage='sqlite:///optimization_' + storage_name + '.db')
    study.optimize(lambda trial: objective(trial, X, static_variable_obj), n_trials=n_trials, catch=(Exception,))


def run_optimization(X, storage_name,
                     study_name,
                     direction=config["optimizer_config"]["frame_work"]["direction"],
                     n_trials=config["optimizer_config"]["frame_work"]["n_trials"],
                     trial_parallel=True,
                     algorithm=None):
    """
    run_optimization utilize optuna framework from auto_AD package
    to get the best hyperparameters

    Parameters:
        X (array): is the data to optimize the hps
        study_name(str): study_name to given for algorithm
        direction (str): direction to minimize or maximize the objective function
        n_trials(int): is number of times the objective function to run
        static_variable_obj(config object): auto_AD config parameters
        trial_parallel (True/False): paralleilzation of trials condition

    Returns:
        Best Hyperparameters of the algorithms
    """
    print(trial_parallel)
    hps = deflt_hps_rng(len(X))["algorithms"][algorithm]
    curve = config["optimizer_config"]["frame_work"]["curve"]
    static_variable_obj = variable_obj(hps, algorithm, curve, n_trials, direction)
    cv = ShuffleSplit(n_splits=config["optimizer_config"]["tune_config"]["kfold"]["n_splits"],
                      test_size=config["optimizer_config"]["tune_config"]["kfold"]["test_size"], random_state=42)
    static_variable_obj.cv = cv
    study = optuna.create_study(study_name=study_name, storage='sqlite:///optimization_' + storage_name + '.db',
                                load_if_exists=True, direction=direction)
    if trial_parallel == False:
        results = Parallel(n_jobs=1)([delayed(optimize)(1, study_name, X,
                                                        storage_name, static_variable_obj) for _ in range(n_trials)])
    else:
        results = Parallel(n_jobs=-1)([delayed(optimize)(1, study_name, X,
                                                         storage_name, static_variable_obj) for _ in range(n_trials)])
    trial = study.best_trial
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return study