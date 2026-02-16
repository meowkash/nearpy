'''
Adaptive Filter Implementations - 
[1] RLS
[2] LMS 
[3] Kalman Filter 
'''
import numpy as np 

def lms_filter(sig, noise_ref, mu=0.01, n_taps=32):
    n_samples = len(sig)
    weights = np.zeros(n_taps)
    output = np.zeros(n_samples)
    error = np.zeros(n_samples)
    
    for i in range(n_taps, n_samples):
        x = noise_ref[i:i-n_taps:-1] # Noise reference (target)
        y = np.dot(weights, x)       # Predicted noise
        e = sig[i] - y               # Error (Ideally noise-free signal)
        weights = weights + 2 * mu * e * x
        output[i] = y
        error[i] = e
    
    return error

def rls_filter(sig, noise_ref, lam=0.99, n_taps=32):
    n_samples = len(sig)
    weights = np.zeros(n_taps)
    P = np.eye(n_taps) * 1.0  # Inverse correlation matrix
    error = np.zeros(n_samples)
    
    for i in range(n_taps, n_samples):
        x = noise_ref[i:i-n_taps:-1]
        Px = P @ x
        gain = Px / (lam + x.T @ Px)
        y = weights.T @ x
        e = sig[i] - y
        weights = weights + gain * e
        P = (P - np.outer(gain, x.T @ P)) / lam
        error[i] = e
    return error

def kalman_filter(sig, Q=1e-5, R=1e-2):
    n_samples = len(sig)
    x_hat = np.zeros(n_samples) # Estimated signal
    P = np.zeros(n_samples)     # Estimation error covariance
    x_hat[0] = sig[0]
    P[0] = 1.0
    
    for k in range(1, n_samples):
        # Prediction
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        
        # Update
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (sig[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
        
    return x_hat