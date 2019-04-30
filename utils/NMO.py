import cv2
import numpy as np
import os
import seaborn as sns
from scipy import ndimage
from scipy.interpolate import CubicSpline

def nmo_correction(cmp, dt, offsets, velocities):
    """
        Performs NMO correction on the given CMP.
        
        The units must be consistent. E.g., if dt is seconds and
        offsets is meters, velocities must be m/s.
        
        Parameters
        ----------
        
        cmp : 2D array
        The CMP gather that we want to correct.
        dt : float
        The sampling interval.
        offsets : 1D array
        An array with the offset of each trace in the CMP.
        velocities : 1D array
        An array with the NMO velocity for each time. Should
        have the same number of elements as the CMP has samples.
        
        Returns
        -------
        
        nmo : 2D array
        The NMO corrected gather.
        
        """
    nmo = np.zeros_like(cmp)
    nsamples = cmp.shape[0]
    times = np.arange(0, nsamples*dt, dt)
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            amplitude = sample_trace(cmp[:, j], t, dt)
            # If the time t is outside of the CMP time range,
            # amplitude will be None.
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo

def reflection_time(t0, x, vnmo):
    """
        Calculate the travel-time of a reflected wave.
        
        Doesn't consider refractions or changes in velocity.
        
        The units must be consistent. E.g., if t0 is seconds and
        x is meters, vnmo must be m/s.
        
        Parameters
        ----------
        
        t0 : float
        The 0-offset (normal incidence) travel-time.
        x : float
        The offset of the receiver.
        vnmo : float
        The NMO velocity.
        
        Returns
        -------
        
        t : float
        The reflection travel-time.
        
        """
    t = np.sqrt(t0**2 + x**2/vnmo**2)
    return t

def sample_trace(trace, time, dt):
    """
        Sample an amplitude at a given time using interpolation.
        
        Parameters
        ----------
        
        trace : 1D array
        Array containing the amplitudes of a single trace.
        time : float
        The time at which I want to sample the amplitude.
        dt : float
        The sampling interval
        
        Returns
        -------
        
        amplitude : float or None
        The interpolated amplitude. Will be None if *time*
        is beyond the end of the trace or if there are less
        than 2 points between *time* and the end.
        
        """
    # The floor function will give us the integer
    # right behind a given float.
    # Use it to get the sample number that is right
    # before our desired time.
    before = int(np.floor(time/dt))
    N = trace.size
    # Use the 4 samples around time to interpolate
    samples = np.arange(before - 1, before + 3)
    if any(samples < 0) or any(samples >= N):
        amplitude = None
    else:
        times = dt*samples
        amps = trace[samples]
        interpolator = CubicSpline(times, amps)
        amplitude = interpolator(time)
    return amplitude
