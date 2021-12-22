sys.path.append("C:\\AutoForecast\\auto_AD")
sys.path.append("C:\\AutoForecast\\auto_AD\\model_factory")
sys.path.append("C:\\AutoForecast\\auto_AD\\misc")
sys.path.append("C:\\AutoForecast\\auto_AD\\ts_transformers")

from auto_AD.model_main import run_algorithm
from auto_AD.optimization_main import run_optimization
from auto_AD.misc.hp_config import deflt_hps
from auto_AD.misc.hp_config import deflt_hps_rng
from auto_AD.misc.misc import setup_logger

import yaml
import warnings
from joblib import parallel,delayed
import numpy as np
import pandas as pd

X = np.random.rand(100,2)
i_logger_1 = setup_logger("auto_AD","auto_AD.log")
algorithms = ["IForest_model","HBOS_model","COPOD_model"
              ,"CBLOF_model"]

# Scaling to parameterized
# Best_hps to check


from ts_transformer import transformer_ts
s = pd.read_pickle("C:\\AutoForecast\\cpu.pickle")

ts_object = transformer_ts(changepoint="changepoint_ADTK",trend="median",seasonality="FFT",n_jobs=1)
res_df = ts_object.decompose(s)


X = np.array(res_df)
model_object = run_algorithm(algorithms,remove_zeros = True,normalization = True,
                             ensemble ="average",n_jobs = 1, hps = None,i_logger = i_logger_1,best_hps = True)
all_algo = model_object.run_ensemble(X)




