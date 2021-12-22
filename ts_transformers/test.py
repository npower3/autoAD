
import pandas as pd
import numpy as np
s = pd.read_csv('./cpu.csv', index_col="Time", parse_dates=True, squeeze=True)
y2 = np.array(s)
segments_prps = changepoint(y2,2,8)
segments_pops = segments_prps.ofcd_processing()

from adtk.detector import LevelShiftAD
from adtk.data import validate_series
from adtk.visualization import plot

s = validate_series(s)
level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
anomalies = level_shift_ad.fit_detect(s)
plot(s, anomaly=anomalies, anomaly_color='red')

# plt.plot(range(len(y2)),y2, 'r')


def generate_normal_time_series(num, minl=50, maxl=700):
    y = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn() * 10
        var = np.random.randn() * 1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        y = np.concatenate((y, tdata))
    return y

y2 = generate_normal_time_series(7, 50, 200)


y2 = pd.DataFrame({"Time":s.index[0:len(y2)],"y":y2})
y2 = y2.set_index("Time")
s = validate_series(y2)
pd.DataFrame(anomalies).to_csv("y.csv")


s = validate_series(s)
level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
anomalies = level_shift_ad.fit_detect(s)
plot(s, anomaly=anomalies, anomaly_color='red')

cp_df = pd.DataFrame(anomalies).loc[pd.DataFrame(anomalies)["CPU (%)"]==1]
cp_df["factor"]  = list(range(0,cp_df.shape[0]))
cp_df["factor"] = [((i+1) % 5) for i in cp_df["factor"]]
cp_df = cp_df[["factor"]]

# cp_df.loc[cp_df["factor"]==0].index.tolist()
# cp_df.loc[cp_df["factor"]==0].index.tolist()

cp_df_1 = pd.merge(pd.DataFrame(s),cp_df,left_index=True, right_index=True,how="left")
cp_df_1.iloc[0,1] = 0
cp_df_1.iloc[-1,1] = 0
segment_inde = cp_df_1[cp_df_1["factor"]==0].index.tolist()
master_inde = cp_df_1.index.tolist()
i = 0
segment = []
while len(segment_inde) > 1:
    start_inde = master_inde.index(segment_inde[0])
    end_inde = master_inde.index(segment_inde[1])
    segment_inde.pop(0)
    inde = cp_df_1.index[start_inde:end_inde]
    if len(segment_inde) == 1:
        inde = cp_df_1.index[start_inde:(end_inde+1)]
    segment.append(cp_df_1.loc[cp_df_1.index.isin(inde)]["CPU (%)"].tolist())
    print(start_inde)
    print(end_inde)

cp_df_1.index[0]
cp_df_1.index[-1]

segment_inde.append(cp_df_1.index[0])
index("bar").append(cp_df_1.index[-1])
inde = cp_df_1.index.tolist()
inde.index(segment_inde[3])



while len(segment_inde)!=0:




window = 5
segments_index = list(range(0,cp_df.shape[0]+window,window))

segments = []
y = np.array(s)
for  segments_ind in  range(0,len(segments_index),1):



cp_df[cp_df.index.isin(cp_df.index[segments_index[segments_ind]])]

cp_df.index[segments_index[0]]
cp_df.index[segments_index[1]]
# cp_df.index[segments_index[2]]



import os
os.chdir("C:\\AutoForecast\\auto_AD\\ts_transformers")
import sys
sys.path.append("C:\\AutoForecast\\auto_AD\\ts_transformers")
from changepoint import *

import pandas as pd
import numpy as np
s = pd.read_csv('C:\\AutoForecast/cpu.csv', index_col="Time", parse_dates=True, squeeze=True)
s = pd.DataFrame(s)
s["CPU (%)"]  = range(0,1000,1)

cp_ts = changepoint_MCMC(s)
cp_ts.detect_cps()



lvl_shift = []
for i in range(0, len(cp_ts.__dict__["segments"])):
    lvl_shift.append(len(cp_ts.__dict__["segments"][i]))
anomalies = np.cumsum(lvl_shift)[:-1]
x = s.index
y = list(s)
plt.plot(x, y, "-o")
x0 = s.index[list(anomalies)]
y0 = s[list(anomalies)].tolist()
plt.plot(x0, y0, "s")
plt.show()

