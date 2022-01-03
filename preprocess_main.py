# import sys
# sys.path.append("C:\\autoAD")
# sys.path.append("C:\\autoAD\\auto_AD\\model_factory")
# sys.path.append("C:\\autoAD\\auto_AD\\misc")
# sys.path.append("C:\\autoAD\\auto_AD\\ts_transformers")
# sys.path.append("C:\\autoAD\\auto_AD\\preprocess")

from joblib import Parallel, delayed
from preprocess.encoder import Encoder
import pandas as pd


# df_flights = pd.read_csv('https://raw.githubusercontent.com/ismayc/pnwflights14/master/data/flights.csv')
# df=df_flights
#
# #### boston data
# # prepare some data
# bunch = load_boston()
# y = bunch.target
# X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
# X['target']=y

def categorical_identifier(X):
    cat_var = X.select_dtypes(include=['object']).copy().columns
    cat_var = ["CHAS","RAD"]            # remove
    cardinality = []
    for cat in cat_var:
        cardinality.append(X[cat].nunique())
    return_df = pd.DataFrame({"cat_feat":cat_var})
    return_df["data_type"] = "Nominal"
    return_df["cardinality"] = cardinality
    return_df["Info_loss"] = "NO"
    return_df["Response_leakage"] = "yes"
    return_df["diff_dat"] = "Equal"
    return_df["contrast_en"] =  "yes"
    return_df["run_id"] = 1
    return return_df

def encode_categorical(cat_df,Tree_based_Algo):

    data_type =  cat_df["data_type"].tolist()[0]
    cardinality = cat_df["cardinality"].tolist()[0]
    Info_loss = cat_df["Info_loss"].tolist()[0]
    Response_leakage = cat_df["Response_leakage"].tolist()[0]
    diff_dat = cat_df["diff_dat"].tolist()[0]
    contrast_en = cat_df["contrast_en"].tolist()[0]
    run_id = cat_df["run_id"].tolist()[0]

    if data_type == "Nominal":
        if cardinality < 15:
            if Tree_based_Algo == "No":
                cat_encode = ['one-hot']
            else:
                if Info_loss == 'No' and Supervised == True:
                    if Response_leakage == 'No':
                        cat_encode = ['CatBoost', 'Leave one out']
                    else:
                        cat_encode = ['Target', 'weight of evidence', 'james-stein', 'M-estimator']
                else:
                    cat_encode = ['Binary', 'Feature Hashing']

        else:
            if Info_loss == 'No' and Supervised == True:
                if Response_leakage == 'No':
                    cat_encode = ['CatBoost', 'Leave one out']
                else:
                    cat_encode = ['Target', 'weight of evidence', 'james-stein', 'M-estimator']
            else:
                cat_encode = ['Binary', 'Feature Hashing']

    if data_type == "Ordinal":
        if diff_dat == 'Equal':
            cat_encode = ['Ordinal']
        else:
            if contrast_en == True:
                cat_encode = ['Helmert', 'Sum', 'backward-difference', 'Polynomial']
            else:
                if cardinality < 15:
                    if Tree_based_Algo == "no":
                        cat_encode = ['one-hot']
                    else:
                        if Info_loss == 'No' and Supervised == True:
                            if Response_leakage == 'No':
                                cat_encode = ['CatBoost', 'Leave one out']
                            else:
                                cat_encode = ['Target', 'weight of evidence', 'james-stein', 'M-estimator']
                        else:
                            cat_encode = ['Binary', 'Feature Hashing']
                else:
                    if Info_loss == 'No' and Supervised == True:
                        if Response_leakage == 'No':
                            cat_encode = ['CatBoost', 'Leave one out']
                        else:
                            cat_encode = ['Target', 'weight of evidence', 'james-stein', 'M-estimator']
                    else:
                        cat_encode = ['Binary', 'Feature Hashing']
    return cat_encode

def myfunc_data(row,X,y=None):
    cols = row["cat_feat"].unique()
    encode_method = row["recommended_method"][0]
    encoder_object = Encoder( method=encode_method, to_encode=cols)
    encoder_object = encoder_object.fit(X, cols)
    X_encode = encoder_object.transform(self, X)
    return X_encode


class auto_encoder:
    def __init__(self,Tree_based_Algo = "no",jobs = 1):
        self.Tree_based_Algo = Tree_based_Algo
        self.jobs = jobs
    def transform(self,X,cat_df=None):
        if cat_df == None:
            cat_df = categorical_identifier(X)
        encode_cat_list = Parallel(n_jobs=self.jobs)(delayed(
            encode_categorical)(cat_df.loc[cat_df["cat_feat"]==cat],
                             self.Tree_based_Algo) for cat in list(cat_df["cat_feat"].unique()))
        cat_df["recommendation"] = encode_cat_list
        cat_df["recommended_method"] = [i[0] for i in cat_df["recommendation"]]
        encode_x_list = cat_df.groupby(["recommended_method"]).apply(lambda row: myfunc_data(row, X))
        encode_x = pd.concat(encode_x_list, axis=1)
        return encode_x






















import sys, os
sys.path.append(os.path.abspath("."))














