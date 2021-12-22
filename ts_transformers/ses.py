
import statsmodels.api as sm

class seasonality_extraction:
    def __init__(self,kind):
        self.kind = kind
    def extractor(self,ts):
        ts_ses = sm.tsa.seasonal_decompose(ts)
        return ts_ses._seasonal

