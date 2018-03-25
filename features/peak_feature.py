from features.feature import WindowFeature
from features.peakdetect import peakdetect
import numpy as np
from scipy import interpolate

class WindowPeakTroughPoints(WindowFeature):
    def __init__(self):
        pass

    def calc_feature(self, window, lookahead=5, delta=0.02):
        '''
            Returns the index of troughs and period between them for the supplied co2 signal.
            Note: Troughs are used over peaks since the troughs are more sharp for the co2 signal.
        '''
        # Only grab the troughs not the peaks
        troughs_idx_value, peaks_idx_value = peakdetect(window, lookahead=lookahead, delta=delta)
        # Just need the position
        troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
        troughs_val = np.asarray([x[1] for x in troughs_idx_value])
        peaks_idx = np.asarray([x[0] for x in peaks_idx_value])
        peaks_val = np.asarray([x[1] for x in peaks_idx_value])

        troughs_period = self._calc_period(troughs_idx)
        peaks_period = self._calc_period(peaks_idx)

        return ((peaks_idx, peaks_val, peaks_period),(troughs_idx, troughs_val, troughs_period))

    def _calc_period(self, idx):
        troughs_period = np.empty_like(idx)

        # For the very last trough assume same period as next one
        troughs_period[:-1] = np.diff(idx)
        troughs_period[-1] = troughs_period[-2]
        return troughs_period

class WindowPeakTroughPeriods(WindowFeature):
    def __init__(self):
        pass

    def calc_feature(self, window, interpolate=True):
        '''
            Returns the index of troughs and period between them for the supplied co2 signal.
            Note: Troughs are used over peaks since the troughs are more sharp for the co2 signal.
        '''
        # Only grab the troughs not the peaks
        troughs_idx_value, peaks_idx_value = peakdetect(window, lookahead=5, delta=0.02)
        # Just need the position
        troughs_idx = np.asarray([x[0] for x in troughs_idx_value])
        peaks_idx = np.asarray([x[0] for x in peaks_idx_value])

        troughs_period = self._calc_period(troughs_idx)
        peaks_period = self._calc_period(peaks_idx)

        if interpolate:
            interp_troughs_period = self._interpolate_spline(troughs_idx, troughs_period, np.arange(window.size))
            interp_peaks_period = self._interpolate_spline(peaks_idx, peaks_period, np.arange(window.size))

            return [interp_peaks_period, interp_troughs_period]
        else:
            return ((peaks_idx, peaks_period),(troughs_idx, troughs_period))

    def _calc_period(self, idx):
        troughs_period = np.empty_like(idx)

        # For the very last trough assume same period as next one
        troughs_period[:-1] = np.diff(idx)
        troughs_period[-1] = troughs_period[-2]
        return troughs_period

    def _interpolate_spline(self, in_idx, in_val, out_idx, s=0):
        tck = interpolate.splrep(in_idx, in_val, s=s)
        return interpolate.splev(out_idx, tck, der=0)

