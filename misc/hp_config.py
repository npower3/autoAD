import optuna
import math

def deflt_hps_rng(dim_X):
    dict = {"algorithms": {
        "IForest_model":  "{'n_estimators': trial.suggest_int('n_estimators',"+str(80)+ ","+ str(100)+"), 'max_samples': trial.suggest_uniform('max_samples',"+str(0.1)+","+ str(1.0)+"),'contamination': 0.1, 'max_features': 1., 'bootstrap': False, 'n_jobs': 1, 'behaviour': 'old','random_state': None}",
        # "CBLOF_model": "{'n_clusters':trial.suggest_int('n_clusters',"+str(2)+ ","+ str(40)+"),'contamination':0.1,'clustering_estimator':None,'alpha':trial.suggest_uniform('alpha',"+str(0.9)+","+str(0.95)+"),'beta':trial.suggest_int('beta',"+str(5.0)+","+str(20)+"),'use_weights':False,'check_estimator':False,'random_state':None,'n_jobs':1}",
        "CBLOF_model": "{'n_clusters':trial.suggest_int('n_clusters',"+str(2)+ ","+ str(40)+"),'contamination':0.1,'clustering_estimator':None,'alpha':0.9,'beta':5,'use_weights':False,'check_estimator':False,'random_state':None,'n_jobs':1}",

        "KNN_model": "{'contamination': 0.1,'n_neighbors':trial.suggest_int('n_neighbors',"+str(5)+","+ str(min(int(0.1*dim_X),100))+")"+",'method':'largest','radius':1.0,'algorithm':'auto','leaf_size':30,'metric':'minkowski','p':2, 'metric_params':None,'n_jobs':1}",
        "HBOS_model":"{'n_bins':trial.suggest_int('n_bins',"+ str(max(math.sqrt(dim_X)-40,5))+","+ str(max(math.sqrt(dim_X)+40,dim_X))+"), 'alpha':0.1, 'tol':0.5, 'contamination':0.1}",
        "hdbscan_model":"{'algorithm':'best', 'alpha':1.0, 'approx_min_span_tree':True,'gen_min_span_tree':False, 'leaf_size':40,'metric':'euclidean', 'min_cluster_size':trial.suggest_int('min_cluster_size',"+ str(10)+","+ str(min(0.1*dim_X,100)) + "), 'min_samples':trial.suggest_int('min_samples',"+str(5)+","+ str(min(0.1*dim_X,100))+"), 'p':None,'prediction_data':True}",
        "LOF_model":"{'n_neighbors':trial.suggest_int('n_neighbors',"+str(5)+","+str(min(int(0.1*dim_X),100))+"),'algorithm':'auto','leaf_size':30,'metric':'minkowski','p':2,'metric_params':None,'contamination':0.1,'n_jobs':1}",
        "COPOD_model":"{}"
    }
    }
    return dict


def deflt_hps():
    dict = {"algorithms": {
        "IForest_model": "{'n_estimators': 100, 'max_samples': 'auto','contamination': 0.1, 'max_features': 1., 'bootstrap': False, 'n_jobs': 1, 'behaviour': 'old','random_state': None}",
        "CBLOF_model": "{'n_clusters':8,'contamination':0.1,'clustering_estimator':None,'alpha':0.9,'beta':5.0,'use_weights':False,'check_estimator':False,'random_state':None,'n_jobs':1}",
        "KNN_model": "{'contamination': 0.1,'n_neighbors':5,'method':'largest','radius':1.0,'algorithm':'auto','leaf_size':30,'metric':'minkowski','p':2, 'metric_params':None,'n_jobs':1}",
        "HBOS_model":"{'n_bins':10, 'alpha':0.1, 'tol':0.5, 'contamination':0.1}",
        "hdbscan_model":"{'algorithm':'best', 'alpha':1.0, 'approx_min_span_tree':True,'gen_min_span_tree':False, 'leaf_size':40,'metric':'euclidean', 'min_cluster_size':5, 'min_samples':None, 'p':None,'prediction_data':True}",
        "LOF_model":"{'n_neighbors':20,'algorithm':'auto','leaf_size':30,'metric':'minkowski','p':2,'metric_params':None,'contamination':0.1,'n_jobs':1}",
        "COPOD_model":"{}"
    }
    }
    return dict
