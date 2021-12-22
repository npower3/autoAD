from ts_transformers.changepoint import changepoint_MCMC
from ts_transformers.changepoint import changepoint_ADTK
from ts_transformers.changepoint import show_plot
from ts_transformers.trend import trend_extraction
from ts_transformers.ses import seasonality_extraction
from itertools import chain
from joblib import Parallel, delayed


def decompose_parallel(ts_i,changepoint,trend,seasonality):
    cp_ts = eval(changepoint)(ts_i).detect_cps()
    segments = cp_ts.segments
    trend_segment = trend_extraction(kind=trend, period=None,
                                     ptimes=2, frequency1=12)
    trend_i = trend_segment.extractor(segments)
    tr = list(chain(*trend_i))
    ts_detrend = ts_i - tr
    seasonality_segment = seasonality_extraction(kind=seasonality)
    ts_ses = seasonality_segment.extractor(ts_detrend)
    residual = ts_detrend - ts_ses
    return residual

class transformer_ts:
    def __init__(self,changepoint="changepoint_MCMC",
                 trend="median",seasonality="FFT",n_jobs=1):
        self.changepoint = changepoint
        self.trend = trend
        self.seasonality = seasonality
        self.n_jobs = n_jobs
    def decompose(self,ts):
        colnames = s.columns.tolist()
        res_list = Parallel(n_jobs=self.n_jobs)(delayed(
            decompose_parallel)(s[colnames[i]],
            changepoint,trend,seasonality) for i in range(0,len(colnames)))
        self.res_df = pd.concat(res_list, axis=1)
        self.res_df.columns = colnames
        return self.res_df






