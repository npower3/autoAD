########## trend extraction algorithms #####################
from __future__ import division
from itertools import chain

import numpy as np
from scipy import stats
import  scipy
MIN_FFT_CYCLES = 3.0

# by default, assume periods of no more than this when establishing
# FFT window sizes.
MAX_FFT_PERIOD = 512

def guess_trended_period(data):
    """return a rough estimate of the major period of trendful data.
    Periodogram wants detrended data to score periods reliably. To do
    that, apply a broad median filter based on a reasonable maximum
    period.  Return a weighted average of the plausible periodicities.
    Parameters
    ----------
    data : ndarray
        list of observed values, evenly spaced in time.
    Returns
    -------
    period : int
    """
    max_period = min(len(data) // 3, 512)
    broad = fit_trend(data, kind="median", period=max_period)
    peaks = periodogram_peaks(data - broad)
    if peaks is None:
        return max_period
    periods, scores, _, _ = zip(*peaks)
    period = int(round(np.average(periods, weights=scores)))
    return period
def periodogram_peaks(data, min_period=4, max_period=None, thresh=0.90):
    """return a list of intervals containg high-scoring periods
    Use a robust periodogram to estimate ranges containing
    high-scoring periodicities in possibly short, noisy data. Returns
    each peak period along with its adjacent bracketing periods from
    the FFT coefficient sequence.
    Data should be detrended for sharpest results, but trended data
    can be accommodated by lowering thresh (resulting in more
    intervals being returned)
    Parameters
    ----------
    data : ndarray
        Data series, evenly spaced samples.
    min_period : int
        Disregard periods shorter than this number of samples.
        Defaults to 4
    max_period : int
        Disregard periods longer than this number of samples.
        Defaults to the smaller of len(data)/MIN_FFT_CYCLES or MAX_FFT_PERIOD
    thresh : float (0..1)
        Retain periods scoring above thresh*maxscore. Defaults to 0.9
    Returns
    -------
    periods : array of quads, or None
        Array of (period, power, period-, period+), maximizing period
        and its score, and FFT periods bracketing the maximizing
        period, returned in decreasing order of score
    """
    periods, power = periodogram(data, min_period, max_period)
    if np.all(np.isclose(power, 0.0)):
        return None # DC
    result = []
    keep = power.max() * thresh
    while True:
        peak_i = power.argmax()
        if power[peak_i] < keep:
            break
        min_period = periods[min(peak_i + 1, len(periods) - 1)]
        max_period = periods[max(peak_i - 1, 0)]
        result.append([periods[peak_i], power[peak_i], min_period, max_period])
        power[peak_i] = 0
    return result if len(result) else None
def fit_trend(data, kind="spline", period=None, ptimes=2):
    """Fit a trend for a possibly noisy, periodic timeseries.
    Trend may be modeled by a line, cubic spline, or mean or median
    filtered series.
    Parameters
    ----------
    data : ndarray
        list of observed values
    kind : string ("mean", "median", "line", "spline", None)
        if mean, apply a period-based mean filter
        if median, apply a period-based median filter
        if line, fit a slope to median-filtered data.
        if spline, fit a piecewise cubic spline to the data
        if None, return zeros
    period : number
        seasonal periodicity, for filtering the trend.
        if None, will be estimated.
    ptimes : number
        multiple of period to use as smoothing window size
    Returns
    -------
    trend : ndarray
    """
    if kind is None:
        return np.zeros(len(data)) + np.mean(data)
    if period is None:
        period = guess_trended_period(data)
    window = (int(period * ptimes) // 2) * 2 - 1 # odd window
    if kind == "median":
        filtered = aglet(median_filter(data, window), window)
    elif kind == "mean":
        filtered = aglet(mean_filter(data, window), window)
    elif kind == "line":
        filtered = line_filter(data, window)
    elif kind == "spline":
        nsegs = len(data) // (window * 2) + 1
        filtered = aglet(spline_filter(data, nsegs), window)
    else:
        raise Exception("adjust_trend: unknown filter type {}".format(kind))
    return filtered

def guess_trended_period(data):
    """return a rough estimate of the major period of trendful data.
    Periodogram wants detrended data to score periods reliably. To do
    that, apply a broad median filter based on a reasonable maximum
    period.  Return a weighted average of the plausible periodicities.
    Parameters
    ----------
    data : ndarray
        list of observed values, evenly spaced in time.
    Returns
    -------
    period : int
    """
    max_period = min(len(data) // 3, 512)
    broad = fit_trend(data, kind="median", period=max_period)
    peaks = periodogram_peaks(data - broad)
    if peaks is None:
        return max_period
    periods, scores, _, _ = zip(*peaks)
    period = int(round(np.average(periods, weights=scores)))
    return period

def aglet(src, window, dst=None):
    """straigten the ends of a windowed sequence.
    Replace the window/2 samples at each end of the sequence with
    lines fit to the full window at each end.  This boundary treatment
    for windowed smoothers is better behaved for detrending than
    decreasing window sizes at the ends.
    Parameters
    ----------
    src : ndarray
        list of observed values
    window : int
        odd integer window size (as would be provided to a windowed smoother)
    dst : ndarray
        if provided, write aglets into the boundaries of this array.
        if dst=src, overwrite ends of src in place. If None, allocate result.
    Returns
    -------
    dst : ndarray
        array composed of src's infield values with aglet ends.
    """
    if dst is None:
        dst = np.array(src)
    half = window // 2
    leftslope = stats.theilslopes(src[: window])[0]
    rightslope = stats.theilslopes(src[-window :])[0]
    dst[0:half] = np.arange(-half, 0) * leftslope + src[half]
    dst[-half:] = np.arange(1, half + 1) * rightslope + src[-half - 1]
    return dst

def median_filter(data, window):
    """Apply a median filter to the data.
    This implementation leaves partial windows at the ends untouched
    """
    filtered = np.copy(data)
    for i in range(window // 2, len(data) - window // 2):
        filtered[i] = np.median(data[max(0, i - window // 2) : i + window // 2 + 1])
    return filtered

def mean_filter(data, window):
    """Apply a windowed mean filter to the data.
    This implementation leaves partial windows at the ends untouched
    """
    filtered = np.copy(data)
    cum = np.concatenate(([0], np.cumsum(data)))
    half = window // 2
    filtered[half : -half] = (cum[window:] - cum[:-window]) / window
    return filtered

def line_filter(data, window):
    """fit a line to the data, after filtering"""
    # knock down seasonal variation with a median filter first
    half = window // 2
    coarse = median_filter(data, window)[half : -half] # discard crazy ends
    slope, _, lower, upper = stats.theilslopes(coarse)
    if lower <= 0.0 and upper >= 0.0:
        filtered = np.zeros(len(data)) + np.median(data)
    else:
        intercept = np.median(data) - (len(data) - 1) / 2 * slope
        filtered = slope * np.arange(len(data)) + intercept
    return filtered

def spline_filter(data, nsegs):
    """Detrend a possibly periodic timeseries by fitting a coarse piecewise
       smooth cubic spline
    Parameters
    ----------
    data : ndarray
        list of observed values
    nsegs : number
        number of spline segments
    Returns
    -------
    filtered : ndarray
    """
    index = np.arange(len(data))
    nknots = max(2, nsegs + 1)
    knots = np.linspace(index[0], index[-1], nknots + 2)[1:-2]
    return LSQUnivariateSpline(index, data, knots)(index)
def periodogram(data, min_period=4, max_period=None):
    """score periodicities by their spectral power.
    Produce a robust periodogram estimate for each possible periodicity
    of the (possibly noisy) data.
    Parameters
    ----------
    data : ndarray
        Data series, having at least three periods of data.
    min_period : int
        Disregard periods shorter than this number of samples.
        Defaults to 4
    max_period : int
        Disregard periods longer than this number of samples.
        Defaults to the smaller of len(data)/MIN_FFT_CYCLES or MAX_FFT_PERIOD
    Returns
    -------
    periods, power : ndarray, ndarray
        Periods is an array of Fourier periods in descending order,
        beginning with the first one greater than max_period.
        Power is an array of spectral power values for the periods
    Notes
    -----
    This uses Welch's method (no relation) of periodogram
    averaging[1]_, which trades off frequency precision for better
    noise resistance. We don't look for sharp period estimates from
    it, as it uses the FFT, which evaluates at periods N, N/2, N/3, ...,
    so that longer periods are sparsely sampled.
    References
    ----------
    .. [1]: https://en.wikipedia.org/wiki/Welch%27s_method
    """
    if max_period is None:
        max_period = int(min(len(data) / MIN_FFT_CYCLES, MAX_FFT_PERIOD))
    nperseg = min(max_period * 2, len(data) // 2) # FFT window
    freqs, power = scipy.signal.welch(
        data, 1.0, scaling='spectrum', nperseg=nperseg)
    periods = np.array([int(round(1.0 / freq)) for freq in freqs[1:]])
    power = power[1:]
    # take the max among frequencies having the same integer part
    idx = 1
    while idx < len(periods):
        if periods[idx] == periods[idx - 1]:
            power[idx-1] = max(power[idx-1], power[idx])
            periods, power = np.delete(periods, idx), np.delete(power, idx)
        else:
            idx += 1
    power[periods == nperseg] = 0 # disregard the artifact at nperseg
    min_i = len(periods[periods >= max_period]) - 1
    max_i = len(periods[periods < min_period])
    periods, power = periods[min_i : -max_i], power[min_i : -max_i]
    return periods, power



class trend_extraction:
    def __init__(self,kind="spline", period=None, ptimes=2,frequency1 = 12):
        self.kind = kind
        self.period =  period
        self.ptimes = ptimes
        self.frequency1 = frequency1
    def extractor(self,segments):
        ###### Trend fit after getting final segments
        trend_i = []
        for i in range(len(segments)):
            if (len(segments[i]) > self.frequency1):
                trend_i.append(list(fit_trend(segments[i], kind = self.kind, ptimes = self.ptimes)))
            else:
                trend_i.append(list(fit_trend(segments[i], kind = self.kind, ptimes = self.ptimes)))
        return trend_i




