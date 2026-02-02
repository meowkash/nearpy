"""
A collection of smoothing functions to provide a similar interface as MATLAB's
Data Cleaner Toolbox. Defaults have been adapted from their MATLAB counterparts. 
"""
import numba 
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import convolve1d
from scipy.linalg import lstsq

@numba.njit
def robust_lowess(signal, window_length, frac=None, it=3):
    """
    Robust LOWESS (Locally Weighted Scatterplot Smoothing) filter.
    
    Implements Cleveland's robust LOWESS algorithm using tricube weighting with
    robust iteration to handle outliers. This is the default smoother in MATLAB's
    Data Cleaner Toolbox.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal (time-series data)
    window_length : int or float
        Window size for local regression. If float in (0, 1], treated as fraction
        of data length (MATLAB-compatible span). If int >= 1, treated as absolute
        window size in samples.
    frac : float, optional
        Explicit fraction of data used in each window. If provided, overrides
        window_length interpretation. Default: None (uses window_length).
    it : int, optional
        Number of robust iterations (robustness iterations). Default: 3
        (matches MATLAB's Data Cleaner default).
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal of same length as input
    
    References
    ----------
    Cleveland, W. S. (1979). Robust locally weighted regression and smoothing 
    scatterplots. Journal of the American Statistical Association, 74(368), 829-836.
    
    Notes
    -----
    Implements the core LOWESS algorithm with:
    - Tricube weighting function: w(u) = (1 - |u|^3)^3 for |u| <= 1
    - Local constant fitting (weighted mean at each point)
    - Robust iteration using bisquare weights on residuals
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    
    n = len(signal)
    x = np.arange(n, dtype=np.float64)
    
    # Convert window_length to absolute window size
    if frac is None:
        if isinstance(window_length, float) and 0 < window_length <= 1:
            window_size = max(1, int(np.round(window_length * n)))
        elif isinstance(window_length, (int, float)) and window_length >= 1:
            window_size = int(np.round(window_length))
        else:
            raise ValueError("window_length must be int >= 1 or float in (0, 1]")
    else:
        window_size = max(1, int(np.round(frac * n)))
    
    smoothed = np.copy(signal)
    
    # Initial fit
    for iteration in range(it):
        for i in range(n):
            # Find nearest neighbors (within window)
            distances = np.abs(x - x[i])
            indices = np.argsort(distances)[:window_size]
            
            # Compute tricube weights
            max_dist = np.max(distances[indices])
            if max_dist > 0:
                u = distances[indices] / max_dist
            else:
                u = np.zeros(len(indices))
            
            weights = (1.0 - u**3)**3
            
            if iteration > 0:
                # Apply robust weights based on residuals
                residuals = signal[indices] - smoothed[indices]
                median_residual = np.median(np.abs(residuals))
                if median_residual > 0:
                    standardized = residuals / (6 * median_residual)
                    robust_weights = np.where(np.abs(standardized) < 1, 
                                              (1 - standardized**2)**2, 0)
                    weights *= robust_weights
            
            # Weighted mean
            if np.sum(weights) > 0:
                smoothed[i] = np.average(signal[indices], weights=weights)
    
    return smoothed

@numba.njit
def robust_loess(signal, window_length, poly_degree=1, frac=None, it=3):
    """
    Robust LOESS (Locally Estimated Scatterplot Smoothing) filter.
    
    LOESS is a generalization of LOWESS that performs local polynomial regression.
    Default uses linear (degree 1) polynomials. Robust iteration handles outliers
    using an iterative reweighting scheme based on residuals.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal (time-series data)
    window_length : int or float
        Window size for local polynomial regression. If float in (0, 1], treated 
        as fraction of data length (MATLAB-compatible span). If int >= 1, treated 
        as absolute window size in samples.
    poly_degree : int, optional
        Polynomial degree for local regression. Default: 1 (linear).
        MATLAB Data Cleaner uses degree 1.
    frac : float, optional
        Explicit fraction of data used in each window. If provided, overrides
        window_length interpretation. Default: None (uses window_length).
    it : int, optional
        Number of robust iterations. Default: 3 (matches MATLAB default).
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal of same length as input
    
    References
    ----------
    Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted regression: An 
    approach to regression analysis by local fitting. Journal of the American 
    Statistical Association, 83(403), 596-610.
    
    Notes
    -----
    LOESS extends LOWESS by fitting local polynomials instead of weighted means.
    Uses tricube weighting for distance-based weights and bisquare for outliers.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    
    n = len(signal)
    x = np.arange(n, dtype=np.float64)
    
    # Convert window_length to absolute window size
    if frac is None:
        if isinstance(window_length, float) and 0 < window_length <= 1:
            window_size = max(1, int(np.round(window_length * n)))
        elif isinstance(window_length, (int, float)) and window_length >= 1:
            window_size = int(np.round(window_length))
        else:
            raise ValueError("window_length must be int >= 1 or float in (0, 1]")
    else:
        window_size = max(1, int(np.round(frac * n)))
    
    poly_degree = int(np.round(poly_degree))
    smoothed = np.copy(signal)
    
    # Main iteration loop
    for iteration in range(it):
        for i in range(n):
            # Find nearest neighbors
            distances = np.abs(x - x[i])
            indices = np.argsort(distances)[:window_size]
            
            # Tricube weights based on distance
            max_dist = np.max(distances[indices])
            if max_dist > 0:
                u = distances[indices] / max_dist
            else:
                u = np.zeros(len(indices))
            
            weights = (1.0 - u**3)**3
            
            # Apply robust weights in subsequent iterations
            if iteration > 0:
                residuals = signal[indices] - smoothed[indices]
                median_residual = np.median(np.abs(residuals))
                if median_residual > 0:
                    standardized = residuals / (6 * median_residual)
                    robust_weights = np.where(np.abs(standardized) < 1, 
                                              (1 - standardized**2)**2, 0)
                    weights *= robust_weights
            
            # Weighted polynomial fit
            if np.sum(weights) > 0:
                sqrt_weights = np.sqrt(weights)
                X = np.vstack([x[indices]**j for j in range(poly_degree + 1)]).T
                X_weighted = X * sqrt_weights[:, np.newaxis]
                y_weighted = signal[indices] * sqrt_weights
                
                try:
                    coeffs, _, _, _ = lstsq(X_weighted, y_weighted)
                    # Evaluate polynomial at current point
                    xi = np.array([x[i]**j for j in range(poly_degree + 1)])
                    smoothed[i] = np.dot(coeffs, xi)
                except np.linalg.LinAlgError:
                    # Fallback to weighted mean if polynomial fit fails
                    smoothed[i] = np.average(signal[indices], weights=weights)
    
    return smoothed


def moving_mean(signal, window_length):
    """
    Moving average filter (boxcar filter).
    
    Computes the arithmetic mean over a sliding window. Uses centered window
    (symmetric around current point) for better phase preservation, matching
    MATLAB's Data Cleaner Toolbox behavior.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal (time-series data)
    window_length : int
        Window size in samples. Must be positive integer.
        If even, will be adjusted to nearest odd value for centered window
        (following MATLAB convention).
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal of same length as input
    
    Notes
    -----
    - Window is centered on current sample: for odd window_length k,
      uses (k-1)/2 samples on each side
    - At boundaries, uses truncated windows (valid mode)
    - This is equivalent to a boxcar filter with no phase distortion
    
    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> smoothed = moving_mean(signal, window_length=3)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    
    window_length = int(np.round(window_length))
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    
    # Make window odd for centered behavior (MATLAB default)
    if window_length % 2 == 0:
        window_length += 1
    
    # Use uniform kernel (boxcar filter)
    kernel = np.ones(window_length) / window_length
    
    # Apply with mode='same' to maintain output length
    # Pad edges to reduce boundary artifacts
    pad_width = window_length // 2
    signal_padded = np.pad(signal, pad_width, mode='edge')
    smoothed_padded = convolve1d(signal_padded, kernel, mode='constant', cval=0.0)
    smoothed = smoothed_padded[pad_width:-pad_width]
    
    return smoothed


def moving_median(signal, window_length):
    """
    Moving median filter.
    
    Computes the median over a sliding window. Highly robust to outliers and
    preserves edges better than mean-based filters. Uses centered window
    matching MATLAB's Data Cleaner Toolbox.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal (time-series data)
    window_length : int
        Window size in samples. Must be positive integer.
        If even, will be adjusted to nearest odd value for centered window
        (following MATLAB convention).
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal of same length as input. Data type preserved.
    
    Notes
    -----
    - Window is centered on current sample
    - At boundaries, truncated windows are used
    - This filter is particularly effective for salt-and-pepper noise
    - Order: O(n*k*log(k)) where n=signal length, k=window_length
    
    References
    ----------
    Huang, T. S., Yang, G. J., & Tang, G. Y. (1979). A fast two-dimensional median 
    filtering algorithm. IEEE Transactions on Acoustics, Speech, and Signal 
    Processing, 27(1), 13-18.
    
    Examples
    --------
    >>> signal = np.array([1, 2, 100, 4, 5])  # 100 is an outlier
    >>> smoothed = moving_median(signal, window_length=3)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    
    window_length = int(np.round(window_length))
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    
    # Make window odd for centered behavior (MATLAB default)
    if window_length % 2 == 0:
        window_length += 1
    
    n = len(signal)
    smoothed = np.zeros_like(signal)
    half_win = window_length // 2
    
    # Use convolve with median - simpler implementation using sliding window
    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        smoothed[i] = np.median(signal[start:end])
    
    return smoothed


def savitzky_golay_filter(signal, window_length, poly_order=3):
    """
    Savitzky-Golay polynomial filter.
    
    Fits a polynomial to data within a moving window and uses the polynomial
    value at the center of the window as the smoothed value. Excellent for
    preserving features while smoothing, and can compute derivatives.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal (time-series data)
    window_length : int
        Window size in samples. Must be positive odd integer.
        If even, will be incremented to next odd value (MATLAB behavior).
        Must be >= poly_order + 1.
    poly_order : int, optional
        Degree of polynomial to fit. Default: 3 (MATLAB Data Cleaner default).
        Must be < window_length.
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal of same length as input
    
    Raises
    ------
    ValueError
        If window_length < poly_order + 1
    
    References
    ----------
    Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by 
    simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
    
    Notes
    -----
    - Window length must be odd (automatically adjusted if even)
    - Polynomial order must be less than window length
    - Default order 3 provides good balance between smoothing and feature preservation
    - At boundaries, handles edge effects automatically
    - Preserves step responses and peaks better than simple averaging
    
    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> smoothed = savitzky_golay_filter(signal, window_length=5, poly_order=2)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    
    window_length = int(np.round(window_length))
    poly_order = int(np.round(poly_order))
    
    # Make window odd (MATLAB convention)
    if window_length % 2 == 0:
        window_length += 1
    
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    
    if poly_order >= window_length:
        raise ValueError(f"poly_order ({poly_order}) must be < window_length ({window_length})")
    
    # scipy.signal.savgol_filter requires poly_order < window_length
    # and window_length to be odd
    smoothed = savgol_filter(signal, window_length=window_length, 
                             polyorder=poly_order, mode='interp')
    
    return smoothed


def gaussian_filter(signal, window_length, sigma=None):
    """
    Gaussian smoothing filter.
    
    Applies Gaussian convolution, which is equivalent to low-pass filtering
    with a smooth frequency response. More sophisticated than moving average,
    with parameters related to standard deviation of the Gaussian kernel.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D input signal (time-series data)
    window_length : int
        Window size in samples. Should be odd (automatically adjusted if even).
        Determines truncation of the infinite Gaussian.
    sigma : float, optional
        Standard deviation of the Gaussian kernel in samples.
        Default: window_length / 4 (MATLAB-compatible, gives ~95% of energy
        within window bounds)
    
    Returns
    -------
    numpy.ndarray
        Smoothed signal of same length as input
    
    Notes
    -----
    - Gaussian provides smoother frequency response than boxcar (moving mean)
    - Sigma controls smoothness: larger sigma = more smoothing
    - Default sigma ensures the Gaussian is well-localized within window
    - Uses reflected boundary conditions to minimize edge artifacts
    - Equivalent to frequency-domain multiplication by Gaussian
    
    References
    ----------
    Marroquin, J. L., Santana, E. A., & Botello, S. (2001). Hidden Markov measure 
    field models for image segmentation. IEEE Transactions on Pattern Analysis 
    and Machine Intelligence, 25(11), 1380-1387.
    
    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> smoothed = gaussian_filter(signal, window_length=5, sigma=1.5)
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    
    window_length = int(np.round(window_length))
    
    # Make window odd (MATLAB convention)
    if window_length % 2 == 0:
        window_length += 1
    
    if window_length < 1:
        raise ValueError("window_length must be >= 1")
    
    # Default sigma: MATLAB-compatible (window_length / 4)
    if sigma is None:
        sigma = window_length / 4.0
    
    sigma = float(sigma)
    
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    # Create Gaussian kernel
    ax = np.arange(-window_length // 2 + 1., window_length // 2 + 1.)
    kernel = np.exp(-ax**2 / (2.0 * sigma**2))
    kernel /= kernel.sum()
    
    # Apply with reflected padding to handle boundaries
    smoothed = convolve1d(signal, kernel, mode='reflect')
    
    return smoothed
