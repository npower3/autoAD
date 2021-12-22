import numpy as np
from scipy.special import gammaln, multigammaln, comb
from decorator import decorator
import matplotlib.pyplot as plt
from functools import partial
from itertools import chain
from adtk.detector import LevelShiftAD
import numpy as np
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.visualization import plot

try:
    xrange
except NameError:
    xrange = range
try:
    from sselogsumexp import logsumexp
except ImportError:
    from scipy.special import logsumexp
    print("Use scipy logsumexp().")
else:
    print("Use SSE accelerated logsumexp().")

def generate_normal_time_series(num, minl=50, maxl=1000):
        y = np.array([], dtype=np.float64)
        partition = np.random.randint(minl, maxl, num)
        for p in partition:
            mean = np.random.randn()*10
            var = np.random.randn()*1
            if var < 0:
                var = var * -1
            tdata = np.random.normal(mean, var, p)
            y = np.concatenate((y, tdata))
        return y

def _dynamic_programming(f, *args, **kwargs):
    if f.data is None:
        f.data = args[0]
    if not np.array_equal(f.data, args[0]):
        f.cache = {}
        f.data = args[0]
    try:
        f.cache[args[1:3]]
    except KeyError:
        f.cache[args[1:3]] = f(*args, **kwargs)
    return f.cache[args[1:3]]

def dynamic_programming(f):
    f.cache = {}
    f.data = None
    return decorator(_dynamic_programming, f)

def offline_changepoint_detection(y, prior_func,
                                  observation_log_likelihood_function,
                                  truncate=-np.inf):
    """Compute the likelihood of changepoints on data.
    Keyword arguments:
    data                                -- the time series data
    prior_func                          -- a function given the likelihood of a changepoint given the distance to the last one
    observation_log_likelihood_function -- a function giving the log likelihood
                                           of a data part
    truncate                            -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood
    P                                   -- the likelihoods if pre-computed
    """

    n = len(y)
    Q = np.zeros((n,))
    g = np.zeros((n,))
    G = np.zeros((n,))
    P = np.ones((n, n)) * -np.inf

    # save everything in log representation
    for t in range(n):
        g[t] = np.log(prior_func(t))
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t-1], g[t])
    P[n-1, n-1] = observation_log_likelihood_function(y, n-1, n)
    Q[n-1] = P[n-1, n-1]

    for t in reversed(range(n-1)):
        P_next_cp = -np.inf  # == log(0)
        for s in range(t, n-1):
            P[t, s] = observation_log_likelihood_function(y, t, s+1)

            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # truncate sum to become approx. linear in time (see
            # Fearnhead, 2006, eq. (3))
            if summand - P_next_cp < truncate:
                break
        P[t, n-1] = observation_log_likelihood_function(y, t, n)
        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n-1-t]))
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n-1-t])
        Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG)

    Pcp = np.ones((n-1, n-1)) * -np.inf
    for t in range(n-1):
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t]):
            Pcp[0, t] = -np.inf
    for j in range(1, n-1):
        for t in range(j, n-1):
            tmp_cond = Pcp[j-1, j-1:t] + P[j:t+1, t] + Q[t + 1] + g[0:t-j+1] - Q[j:t+1]
            Pcp[j, t] = logsumexp(tmp_cond.astype(np.float32))
            if np.isnan(Pcp[j, t]):
                Pcp[j, t] = -np.inf
    return Q, P, Pcp

@dynamic_programming
def gaussian_obs_log_likelihood(y, t, s):
    s += 1
    n = s - t
    mean = y[t:s].sum(0) / n
    muT = (n * mean) / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = 1 + 0.5 * ((y[t:s] - mean) ** 2).sum(0) + ((n)/(1 + n)) * (mean**2 / 2)
    scale = (betaT*(nuT + 1))/(alphaT * nuT)
    # splitting the PDF of the student distribution up is /much/ faster.
    # (~ factor 20) using sum over for loop is even more worthwhile
    prob = np.sum(np.log(1 + (y[t:s] - muT)**2/(nuT * scale)))
    lgA = gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT/2)
    return np.sum(n * lgA - (nuT + 1)/2 * prob)
def const_prior(r, l):
    return 1/(l)

def show_plot(cp_ts,s):
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


class changepoint_MCMC:

    def __init__(self,y_ts,cut_off=2,diff_min=8):
        self.y = np.array(y_ts)
        self.cut_off = cut_off
        self.diff_min = diff_min
    def detect_cps(self):
        Q, P, Pcp = offline_changepoint_detection(self.y, partial(const_prior, l=(len(self.y) + 1)), gaussian_obs_log_likelihood,
                                                  truncate=-40)
        ns = np.exp(Pcp).sum(0)
        ###### extarction of anomalies from changepoint algo to get changepoints
        upper = np.where(ns > ns.mean() + self.cut_off * ns.std())
        ######## Creating segments based on changepoints
        segments = []
        seg_med = []
        seg_len = []
        ###### pls add exceptions for when upper is empty
        a11 = len(upper[0])
        upper_re = upper[0]
        y2 = self.y
        for i in range(len(upper_re) + 1):
            if i == 0:
                segments.append(list(y2[0:(upper_re[i] + 1)]))
                seg_med.append(np.median(y2[0:(upper_re[i] + 1)]))
                seg_len.append(len(y2[0:(upper_re[i] + 1)]))
            elif i == a11:
                segments.append(list(y2[(upper_re[(i - 1)] + 1):len(y2)]))
                seg_med.append(np.median(y2[(upper_re[(i - 1)] + 1):len(y2)]))
                seg_len.append(len(y2[(upper_re[(i - 1)] + 1):len(y2)]))
            else:
                segments.append(list(y2[(upper_re[(i - 1)] + 1):(upper_re[i] + 1)]))
                seg_med.append(np.median(y2[(upper_re[(i - 1)] + 1):(upper_re[i] + 1)]))
                seg_len.append(len(y2[(upper_re[(i - 1)] + 1):(upper_re[i] + 1)]))
        ######## elimination of change points if between two consecutive points distance less than diff_min
        ##### using level change ratio of segments to mix the shorter one with their neighbours
        short_cp = list(np.where(np.array(seg_len) <= self.diff_min))
        while len(short_cp[0]) >= 1 and short_cp[0][0] != (len(seg_len) - 1):
            i = short_cp[0][0]
            if ((np.median(segments[(i + 1)]) / np.median(segments[(i - 1)])) > 1) and (
            (np.median(segments[(i + 1)]) / np.median(list(chain(segments[(i - 1)], segments[i]))))) > (
            (np.median(list(chain(segments[(i)], segments[i + 1]))) / np.median(segments[(i - 1)]))):
                segments[(i - 1)] = np.array(list(chain(*[segments[(i - 1)], segments[i]])))
                seg_len[(i - 1)] = len(segments[(i - 1)])
            elif ((np.median(segments[(i + 1)]) / np.median(segments[(i - 1)])) < 1) and (
            (np.median(segments[(i + 1)]) / np.median(list(chain(segments[(i - 1)], segments[i]))))) < (
            (np.median(list(chain(segments[(i)], segments[i + 1]))) / np.median(segments[(i - 1)]))):
                segments[(i - 1)] = np.array(list(chain(*[segments[(i - 1)], segments[i]])))
                seg_len[(i - 1)] = len(segments[(i - 1)])
            else:
                segments[(i + 1)] = np.array(list(chain(*[segments[(i)], segments[(i + 1)]])))
                seg_len[(i + 1)] = len(segments[(i + 1)])
            del segments[i]
            seg_len[i] = 0
            seg_len = [ele for ele in seg_len if ele != 0]
            short_cp = list(np.where(np.array(seg_len) <= self.diff_min))
            # if len(short_cp[0])==0:
        if len(short_cp[0]) >= 1:
            if short_cp[0][0] == (len(seg_len) - 1):
                segments[(i - 1)] = np.array(list(chain(*[segments[(i - 1)], segments[i]])))
                del segments[i]
                seg_len[i] = 0
                seg_len = [ele for ele in seg_len if ele != 0]
        cp = []
        a = 0
        for i in range(len(segments)):
            a = a + len(segments[i])
            cp.append(a)
        cp = cp[0:(len(cp) - 1)]
        ##### After elimination of change points getting proper segments for trend fits
        segments = []
        a11 = len(cp)
        upper_re = cp
        for i in range(len(upper_re) + 1):
            if i == 0:
                segments.append(y2[0:(upper_re[i])])
            elif i == a11:
                segments.append(y2[(upper_re[(i - 1)]):len(y2)])
            else:
                segments.append(y2[(upper_re[(i - 1)]):(upper_re[i])])
        j = 0
        upper_b = 1.5
        lower_b = .5
        while len(segments) > 1 and j < (len(segments) - 1):
            seg_ratio = np.median(segments[(j + 1)]) / np.median(segments[j])
            if seg_ratio > upper_b or seg_ratio < lower_b:
                segments[j] = segments[j]
                j = j + 1
            else:
                segments[j] = np.array(list(chain(*[segments[(j)]], segments[(j + 1)])))
                del segments[(j + 1)]

        cp = []
        a = 0
        for i in range(len(segments)):
            a = a + len(segments[i])
            cp.append(a)

        cp = cp[0:(len(cp) - 1)]
        #### added one more if condition on 20th Dec 2021
        segments = []
        a11 = len(cp)
        upper_re = cp
        if len(upper_re) == 0:
            segments.append(y2)
        else:
            for i in range(len(upper_re) + 1):
                if i == 0:
                    segments.append(y2[0:(upper_re[i])])
                elif i == a11:
                    segments.append(y2[(upper_re[(i - 1)]):len(y2)])
                else:
                    segments.append(y2[(upper_re[(i - 1)]):(upper_re[i])])
        self.segments = segments
        return self



class changepoint_ADTK:
    def __init__(self,y_ts,c=6.0, side='both', window=5):
        self.y = y_ts
        self.c = c
        self.side = side
        self.window = window
    def detect_cps(self):
        level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
        anomalies = level_shift_ad.fit_detect(s)
        cp_df = pd.DataFrame(anomalies).loc[pd.DataFrame(anomalies)["CPU (%)"] == 1]
        cp_df["factor"] = list(range(0, cp_df.shape[0]))
        cp_df["factor"] = [((i + 1) % 5) for i in cp_df["factor"]]
        cp_df = cp_df[["factor"]]
        cp_df_1 = pd.merge(pd.DataFrame(s), cp_df, left_index=True, right_index=True, how="left")
        cp_df_1.iloc[0, 1] = 0
        cp_df_1.iloc[-1, 1] = 0
        segment_inde = cp_df_1[cp_df_1["factor"] == 0].index.tolist()
        master_inde = cp_df_1.index.tolist()
        i = 0
        segment = []
        while len(segment_inde)> 1:
            start_inde = master_inde.index(segment_inde[0])
            end_inde = master_inde.index(segment_inde[1])
            inde = cp_df_1.index[start_inde:(end_inde)]
            print(segment_inde)
            print((start_inde))
            print((end_inde))
            if len(segment_inde) == 2:
                inde = cp_df_1.index[start_inde:(1+end_inde)]
            segment_array = np.asarray(cp_df_1.loc[cp_df_1.index.isin(inde)]["CPU (%)"].tolist())
            segment.append(segment_array)
            segment_inde.pop(0)
        self.segments = segment











