#%% Inverse problem 
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class PVLoopReconstructor:
    def __init__(self, reference_pressure, reference_volume, cycle_length=None):
        """
        Initialize with reference P-V loop data
        
        Parameters:
        -----------
        reference_pressure: np.array
            Pressure values of the reference loop
        reference_volume: np.array
            Volume values of the reference loop
        cycle_length: int, optional
            Number of points in one cardiac cycle
        """
        # Store reference data
        self.ref_p = reference_pressure
        self.ref_v = reference_volume
        
        # If cycle length not provided, assume the entire array is one cycle
        self.cycle_length = cycle_length or len(reference_pressure)
        
        # Create interpolator for the reference P-V relationship
        self.ref_loop = self._create_loop_interpolator()
        
    def _create_loop_interpolator(self):
        """
        Creates an interpolator that can give us volume for any pressure value
        considering the phase of the cardiac cycle
        """
        # Normalize time to [0, 1]
        t = np.linspace(0, 1, self.cycle_length)
        
        # Create separate interpolators for systole and diastole
        # Find approximate end-systolic point (maximum pressure)
        es_idx = np.argmax(self.ref_p)
        
        # Systolic phase (0 to es_idx)
        sys_t = t[:es_idx]
        sys_p = self.ref_p[:es_idx]
        sys_v = self.ref_v[:es_idx]
        
        # Diastolic phase (es_idx to end)
        dia_t = t[es_idx:]
        dia_p = self.ref_p[es_idx:]
        dia_v = self.ref_v[es_idx:]
        
        # Create interpolators
        self.sys_interp_v = interp1d(sys_p, sys_v, bounds_error=False, fill_value="extrapolate")
        self.dia_interp_v = interp1d(dia_p, dia_v, bounds_error=False, fill_value="extrapolate")
        
        return {"systole": self.sys_interp_v, "diastole": self.dia_interp_v}
    
    def _objective_function(self, params, new_pressure, target_loop_area):
        """
        Objective function for optimization
        
        Parameters:
        -----------
        params: array-like
            Parameters for volume scaling and shifting
        new_pressure: array-like
            New pressure trace to match
        target_loop_area: float
            Area of the reference P-V loop
        """
        scale, shift = params
        
        # Generate trial volume trace
        trial_volume = self._generate_volume_trace(new_pressure, scale, shift)
        
        # Calculate loop area
        trial_area = self._calculate_loop_area(new_pressure, trial_volume)
        
        # Calculate error terms
        area_error = (trial_area - target_loop_area)**2
        
        # Add constraints as penalty terms
        volume_range_penalty = 0
        if np.max(trial_volume) > np.max(self.ref_v) * 1.5 or np.min(trial_volume) < np.min(self.ref_v) * 0.5:
            volume_range_penalty = 1e6
        
        return area_error + volume_range_penalty
    
    def _generate_volume_trace(self, pressure, scale=1.0, shift=0.0):
        """Generate volume trace based on pressure and parameters"""
        # Find the phase transition (max pressure point)
        es_idx = np.argmax(pressure)
        
        # Generate volumes for systole and diastole
        vol_sys = self.sys_interp_v(pressure[:es_idx])
        vol_dia = self.dia_interp_v(pressure[es_idx:])
        
        # Combine and apply scaling and shift
        volume = np.concatenate([vol_sys, vol_dia])
        return volume * scale + shift
    
    def _calculate_loop_area(self, pressure, volume):
        """Calculate the area of a P-V loop"""
        return abs(np.trapz(pressure, volume))
    
    def reconstruct_volume(self, new_pressure):
        """
        Reconstruct volume trace from new pressure trace
        
        Parameters:
        -----------
        new_pressure: np.array
            New pressure trace to match with reference loop
        
        Returns:
        --------
        np.array: Reconstructed volume trace
        """
        # Calculate reference loop area
        target_area = self._calculate_loop_area(self.ref_p, self.ref_v)
        
        # Initial guess for parameters
        initial_params = [1.0, 0.0]  # [scale, shift]
        
        # Optimize to match loop characteristics
        result = minimize(
            self._objective_function,
            initial_params,
            args=(new_pressure, target_area),
            method='Nelder-Mead'
        )
        
        # Generate final volume trace
        reconstructed_volume = self._generate_volume_trace(
            new_pressure, 
            scale=result.x[0], 
            shift=result.x[1]
        )
        
        return reconstructed_volume
    
    def plot_results(self, new_pressure, reconstructed_volume):
        """Plot original and reconstructed P-V loops"""
        plt.figure(figsize=(12, 5))
        
        # Original loop
        plt.subplot(121)
        plt.plot(self.ref_v, self.ref_p, 'b-', label='Reference Loop')
        plt.xlabel('Volume (mL)')
        plt.ylabel('Pressure (mmHg)')
        plt.title('Reference P-V Loop')
        plt.grid(True)
        
        # Reconstructed loop
        plt.subplot(122)
        plt.plot(reconstructed_volume, new_pressure, 'r-', label='Reconstructed Loop')
        plt.xlabel('Volume (mL)')
        plt.ylabel('Pressure (mmHg)')
        plt.title('Reconstructed P-V Loop')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
import pandas as pd 

def plot_waveforms_wiggers_style(waveforms):
    """
    Plot the simulated pressure and volume waveforms in Wiggers diagram style.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot pressure and volume waveforms
    ax1.plot(waveforms['t'], waveforms['pressure'], 'r-', linewidth=2, label='LV Pressure')
    ax1.set_ylabel('LVP (mmHg)')
    ax1.set_ylim(-10, 140)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Pressure-Volume Dynamics')
    
    # Add second y-axis for volume
    ax1_twin = ax1.twinx()
    ax1_twin.plot(waveforms['t'], waveforms['volume'], 'k-', linewidth=2, label='LV Volume')
    ax1_twin.set_ylabel('LV Vol (ml)')
    
    # Add cardiac phases
    ymin, ymax = ax1.get_ylim()
    ax1.axvspan(0, 0.065, alpha=0.2, color='gray', label='IVC')
    ax1.axvspan(0.299, 0.38, alpha=0.2, color='gray', label='IVR')
    
    # Add labels for key events
    # ax1.text(0.02, 30, 'Mitral\nValve\nClosing', fontsize=8)
    ax1.text(-0.035, 90, 'Aortic\nValve\nOpening', fontsize=12, fontweight='bold')
    # ax1.text(0.31, 60, 'Aortic\nValve\nClosing', fontsize=8)
    ax1.text(0.43, 5, 'Mitral\nValve\nOpening', fontsize=12, fontweight='bold')
    
    # Plot P-V loop
    ax2.plot(waveforms['volume'], waveforms['pressure'], 'k-', linewidth=2)
    ax2.set_xlabel('LV Vol (ml)')
    ax2.set_ylabel('LVP (mmHg)')
    ax2.set_ylim(-10, 150)
    ax2.set_xlim(0, 150)
    ax2.set_title('Pressure-Volume Loop')
    ax2.grid(True, alpha=0.3)
    
    # Add labels for key points on P-V loop
    ax2.text(waveforms['volume'][0]-5, waveforms['pressure'][0], 'EDV', fontsize=8)
    min_vol_idx = np.argmin(waveforms['volume'])
    ax2.text(waveforms['volume'][min_vol_idx]-5, waveforms['pressure'][min_vol_idx], 'ESV', fontsize=8)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate sample reference data
    t = np.linspace(0, 1, 100)
    t_interp = np.linspace(0, 1, 1000)
    reference_df = pd.DataFrame({
        't': t_interp
    })
    prs_wavefrms = pd.read_csv('Pressure.csv', names=['t', 'pressure'])
    vol_wavefrms = pd.read_csv('Volume.csv', names=['t', 'volume'])

    prs_wavefrms.apply(lambda x: np.round(x, 3))
    vol_wavefrms.apply(lambda x: np.round(x, 3))
    from scipy.interpolate import pchip_interpolate

    ref_volume = pchip_interpolate(vol_wavefrms['t'], vol_wavefrms['volume'], t_interp)
    ref_pressure = pchip_interpolate(prs_wavefrms['t'], prs_wavefrms['pressure'], t_interp)

    reference_df['pressure'] = ref_pressure
    reference_df['volume'] = ref_volume
    
    rescaler = lambda x, l, u: (u-l) * ((x - min(x)) / (max(x) - min(x))) + l 
    
    # Create new pressure trace (slightly different from reference) 
    from nearpy import read_mat
    from scipy.signal import savgol_filter

    matdata = read_mat('C:/Users/Aakash/Desktop/RHC-Study/rv_sub1.mat', legacy=True)
    new_pressure = np.squeeze(matdata['RV_Templ'])
    new_pressure = savgol_filter(new_pressure, window_length=25, polyorder=2)
    new_pressure = rescaler(new_pressure, 0, 120)
    
    # Reconstruct volume
    reconstructor = PVLoopReconstructor(ref_pressure, ref_volume)
    reconstructed_volume = reconstructor.reconstruct_volume(new_pressure)
    
    # Plot results
    reconstructed_df = {
        't': t_interp, 
        'pressure': new_pressure, 
        'volume': reconstructed_volume
    }
    
    plot_waveforms_wiggers_style(reference_df)
    plot_waveforms_wiggers_style(reconstructed_df)
        
#%% Version 2
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from dtaidistance import dtw
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class RobustPVReconstructor:
    def __init__(self, ref_pressure, ref_volume, num_control_points=8):
        """
        Initialize reconstructor with reference data
        
        Parameters:
        -----------
        ref_pressure: np.array
            Reference pressure trace
        ref_volume: np.array
            Reference volume trace
        num_control_points: int
            Number of control points for volume curve reconstruction
        """
        self.ref_p = ref_pressure
        self.ref_v = ref_volume
        self.num_points = len(ref_pressure)
        self.num_control = num_control_points
        
        # Normalize reference signals
        self.ref_p_norm = self._normalize_signal(ref_pressure)
        self.ref_v_norm = self._normalize_signal(ref_volume)
        
        # Extract key features from reference volume
        self.ref_features = self._extract_volume_features(ref_volume)
        
    def _normalize_signal(self, signal):
        """Normalize signal to [0,1] range"""
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    def _extract_volume_features(self, volume):
        """Extract key features from volume trace"""
        # Find key points in the volume curve
        max_idx = np.argmax(volume)
        min_idx = np.argmin(volume)
        
        # Calculate key timing ratios
        ejection_ratio = (min_idx - max_idx) / len(volume)
        filling_ratio = (len(volume) - (min_idx - max_idx)) / len(volume)
        
        return {
            'max_vol': np.max(volume),
            'min_vol': np.min(volume),
            'ejection_ratio': ejection_ratio,
            'filling_ratio': filling_ratio
        }
    
    def _align_signals(self, test_pressure):
        """Perform DTW alignment between reference and test pressure"""
        # Normalize test pressure
        test_p_norm = self._normalize_signal(test_pressure)
        
        # Compute DTW path
        d, path = dtw.warping_paths(self.ref_p_norm, test_p_norm)
        path = dtw.best_path(d)
        
        return path
    
    def _generate_smooth_volume(self, control_points, num_points):
        """Generate smooth volume curve from control points"""
        # Create time points for control points
        t_control = np.linspace(0, 1, len(control_points))
        
        # Create spline interpolator
        cs = CubicSpline(t_control, control_points, bc_type='periodic')
        
        # Generate smooth curve
        t_full = np.linspace(0, 1, num_points)
        volume = cs(t_full)
        
        # Apply additional smoothing
        volume = gaussian_filter1d(volume, sigma=2)
        
        return volume
    
    def _objective_function(self, params, test_pressure, dtw_path):
        """
        Objective function for optimization
        
        Parameters:
        -----------
        params: array-like
            Control points for volume curve
        test_pressure: array-like
            Test pressure trace
        dtw_path: array-like
            DTW alignment path between reference and test pressure
        """
        # Generate smooth volume curve
        volume = self._generate_smooth_volume(params, len(test_pressure))
        
        # Calculate errors
        errors = []
        
        # 1. P-V loop area error
        ref_area = np.trapz(self.ref_p, self.ref_v)
        test_area = np.trapz(test_pressure, volume)
        area_error = ((test_area - ref_area) / ref_area) ** 2
        errors.append(5 * area_error)
        
        # 2. Volume features error
        vol_features = self._extract_volume_features(volume)
        feature_error = (
            ((vol_features['max_vol'] - self.ref_features['max_vol']) / self.ref_features['max_vol']) ** 2 +
            ((vol_features['min_vol'] - self.ref_features['min_vol']) / self.ref_features['min_vol']) ** 2 +
            (vol_features['ejection_ratio'] - self.ref_features['ejection_ratio']) ** 2 +
            (vol_features['filling_ratio'] - self.ref_features['filling_ratio']) ** 2
        )
        errors.append(2 * feature_error)
        
        # 3. Smoothness penalty
        smoothness = np.sum(np.diff(volume, 2) ** 2)
        errors.append(0.1 * smoothness)
        
        # 4. DTW alignment error
        aligned_ref_v = self.ref_v[dtw_path[:,0]]
        aligned_vol = volume[dtw_path[:,1]]
        alignment_error = np.mean((self._normalize_signal(aligned_ref_v) - 
                                 self._normalize_signal(aligned_vol)) ** 2)
        errors.append(3 * alignment_error)
        
        return sum(errors)
    
    def reconstruct_volume(self, test_pressure):
        """
        Reconstruct volume trace from test pressure
        
        Parameters:
        -----------
        test_pressure: np.array
            Test pressure trace to match with reference loop
        
        Returns:
        --------
        np.array: Reconstructed volume trace
        """
        # Perform DTW alignment
        dtw_path = self._align_signals(test_pressure)
        
        # Initial guess for control points based on reference volume
        indices = np.linspace(0, len(self.ref_v)-1, self.num_control, dtype=int)
        initial_control_points = self.ref_v[indices]
        
        # Define bounds for control points
        bounds = [(self.ref_features['min_vol'] * 0.8, self.ref_features['max_vol'] * 1.2)
                 for _ in range(self.num_control)]
        
        # Optimize
        result = minimize(
            self._objective_function,
            initial_control_points,
            args=(test_pressure, dtw_path),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Generate final volume trace
        reconstructed_volume = self._generate_smooth_volume(result.x, len(test_pressure))
        
        return reconstructed_volume
    
    def plot_results(self, test_pressure, reconstructed_volume):
        """Plot original and reconstructed signals and P-V loops"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot pressure traces
        ax1.plot(self.ref_p, 'b-', label='Reference')
        ax1.plot(test_pressure, 'r-', label='Test')
        ax1.set_title('Pressure Traces')
        ax1.legend()
        
        # Plot volume traces
        ax2.plot(self.ref_v, 'b-', label='Reference')
        ax2.plot(reconstructed_volume, 'r-', label='Reconstructed')
        ax2.set_title('Volume Traces')
        ax2.legend()
        
        # Plot P-V loops
        ax3.plot(self.ref_v, self.ref_p, 'b-', label='Reference Loop')
        ax3.set_title('Reference P-V Loop')
        ax3.set_xlabel('Volume (mL)')
        ax3.set_ylabel('Pressure (mmHg)')
        
        ax4.plot(reconstructed_volume, test_pressure, 'r-', label='Reconstructed Loop')
        ax4.set_title('Reconstructed P-V Loop')
        ax4.set_xlabel('Volume (mL)')
        ax4.set_ylabel('Pressure (mmHg)')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # # Generate example data
    # t = np.linspace(0, 1, 100)
    # ref_p = 120 * np.sin(2*np.pi*t)**2 + 10
    # ref_v = 50 * np.cos(2*np.pi*t) + 100
    
    # # Create slightly warped test pressure
    # t_warped = t + 0.1*np.sin(2*np.pi*t)
    # test_p = 110 * np.sin(2*np.pi*t_warped)**2 + 15
    
    # # Reconstruct volume
    # reconstructor = RobustPVReconstructor(ref_p, ref_v)
    # reconstructed_v = reconstructor.reconstruct_volume(test_p)
    
    # # Plot results
    # reconstructor.plot_results(test_p, reconstructed_v)
    
    # Generate sample reference data
    t = np.linspace(0, 1, 100)
    t_interp = np.linspace(0, 1, 1000)
    reference_df = pd.DataFrame({
        't': t_interp
    })
    prs_wavefrms = pd.read_csv('Pressure.csv', names=['t', 'pressure'])
    vol_wavefrms = pd.read_csv('Volume.csv', names=['t', 'volume'])

    prs_wavefrms.apply(lambda x: np.round(x, 3))
    vol_wavefrms.apply(lambda x: np.round(x, 3))
    from scipy.interpolate import pchip_interpolate

    ref_volume = pchip_interpolate(vol_wavefrms['t'], vol_wavefrms['volume'], t_interp)
    ref_pressure = pchip_interpolate(prs_wavefrms['t'], prs_wavefrms['pressure'], t_interp)

    reference_df['pressure'] = ref_pressure
    reference_df['volume'] = ref_volume
    
    rescaler = lambda x, l, u: (u-l) * ((x - min(x)) / (max(x) - min(x))) + l 
    
    # Create new pressure trace (slightly different from reference) 
    from nearpy import read_mat
    from scipy.signal import savgol_filter

    matdata = read_mat('C:/Users/Aakash/Desktop/RHC-Study/rv_sub1.mat', legacy=True)
    new_pressure = np.squeeze(matdata['RV_Templ'])
    new_pressure = savgol_filter(new_pressure, window_length=25, polyorder=2)
    new_pressure = rescaler(new_pressure, 0, 120)
    
    # Reconstruct volume
    reconstructor = RobustPVReconstructor(ref_pressure, ref_volume)
    reconstructed_volume = reconstructor.reconstruct_volume(new_pressure)
    
    # Plot results
    reconstructed_df = {
        't': t_interp, 
        'pressure': new_pressure, 
        'volume': reconstructed_volume
    }
    
    plot_waveforms_wiggers_style(reference_df)
    plot_waveforms_wiggers_style(reconstructed_df)
    
#%% 
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class RobustPVReconstructor:
    def __init__(self, ref_pressure, ref_volume, num_control_points=8):
        """
        Initialize reconstructor with reference data
        
        Parameters:
        -----------
        ref_pressure: np.array
            Reference pressure trace
        ref_volume: np.array
            Reference volume trace
        num_control_points: int
            Number of control points for volume curve reconstruction
        """
        self.ref_p = ref_pressure
        self.ref_v = ref_volume
        self.num_points = len(ref_pressure)
        self.num_control = num_control_points
        
        # Normalize reference signals
        self.ref_p_norm = self._normalize_signal(ref_pressure)
        self.ref_v_norm = self._normalize_signal(ref_volume)
        
        # Extract key features from reference volume
        self.ref_features = self._extract_volume_features(ref_volume)
        
    def _normalize_signal(self, signal):
        """Normalize signal to [0,1] range"""
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    def _extract_volume_features(self, volume):
        """Extract key features from volume trace"""
        # Find key points in the volume curve
        max_idx = np.argmax(volume)
        min_idx = np.argmin(volume)
        
        # Calculate key timing ratios
        ejection_ratio = (min_idx - max_idx) / len(volume)
        filling_ratio = (len(volume) - (min_idx - max_idx)) / len(volume)
        
        return {
            'max_vol': np.max(volume),
            'min_vol': np.min(volume),
            'ejection_ratio': ejection_ratio,
            'filling_ratio': filling_ratio,
            'max_idx': max_idx,
            'min_idx': min_idx
        }
    
    def _align_signals(self, test_pressure):
        """
        Perform DTW alignment between reference and test pressure
        Returns indices mapping between reference and test signals
        """
        # Normalize test pressure
        test_p_norm = self._normalize_signal(test_pressure)
        
        # Compute DTW path
        distance, path = fastdtw(self.ref_p_norm, test_p_norm, dist=euclidean)
        
        # Convert path to numpy array
        path = np.array(path)
        
        return path
    
    def _generate_smooth_volume(self, control_points, num_points):
        """Generate smooth volume curve from control points"""
        # Create time points for control points
        t_control = np.linspace(0, 1, len(control_points))
        
        # Create spline interpolator
        cs = CubicSpline(t_control, control_points, bc_type='periodic')
        
        # Generate smooth curve
        t_full = np.linspace(0, 1, num_points)
        volume = cs(t_full)
        
        # Apply additional smoothing
        volume = gaussian_filter1d(volume, sigma=2)
        
        return volume
    
    def _objective_function(self, params, test_pressure, dtw_path):
        """
        Objective function for optimization
        """
        # Generate smooth volume curve
        volume = self._generate_smooth_volume(params, len(test_pressure))
        
        # Calculate errors
        errors = []
        
        # 1. P-V loop area error
        ref_area = np.trapz(self.ref_p, self.ref_v)
        test_area = np.trapz(test_pressure, volume)
        area_error = ((test_area - ref_area) / ref_area) ** 2
        errors.append(5 * area_error)
        
        # 2. Volume features error
        vol_features = self._extract_volume_features(volume)
        feature_error = (
            ((vol_features['max_vol'] - self.ref_features['max_vol']) / self.ref_features['max_vol']) ** 2 +
            ((vol_features['min_vol'] - self.ref_features['min_vol']) / self.ref_features['min_vol']) ** 2 +
            (vol_features['ejection_ratio'] - self.ref_features['ejection_ratio']) ** 2 +
            (vol_features['filling_ratio'] - self.ref_features['filling_ratio']) ** 2
        )
        errors.append(2 * feature_error)
        
        # 3. Smoothness penalty
        smoothness = np.sum(np.diff(volume, 2) ** 2)
        errors.append(0.1 * smoothness)
        
        # 4. Phase alignment error based on key points
        phase_error = (
            (np.argmax(volume) / len(volume) - self.ref_features['max_idx'] / len(self.ref_v)) ** 2 +
            (np.argmin(volume) / len(volume) - self.ref_features['min_idx'] / len(self.ref_v)) ** 2
        )
        errors.append(3 * phase_error)
        
        return sum(errors)
    
    def reconstruct_volume(self, test_pressure):
        """
        Reconstruct volume trace from test pressure
        """
        try:
            # Perform DTW alignment
            dtw_path = self._align_signals(test_pressure)
            
            # Initial guess for control points based on reference volume
            indices = np.linspace(0, len(self.ref_v)-1, self.num_control, dtype=int)
            initial_control_points = self.ref_v[indices]
            
            # Define bounds for control points
            bounds = [(self.ref_features['min_vol'] * 0.8, self.ref_features['max_vol'] * 1.2)
                     for _ in range(self.num_control)]
            
            # Optimize
            result = minimize(
                self._objective_function,
                initial_control_points,
                args=(test_pressure, dtw_path),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if not result.success:
                print("Warning: Optimization did not converge. Results may be suboptimal.")
            
            # Generate final volume trace
            reconstructed_volume = self._generate_smooth_volume(result.x, len(test_pressure))
            
            return reconstructed_volume
            
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}")
            raise
    
    def plot_results(self, test_pressure, reconstructed_volume):
        """Plot original and reconstructed signals and P-V loops"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot pressure traces
        ax1.plot(self.ref_p, 'b-', label='Reference')
        ax1.plot(test_pressure, 'r-', label='Test')
        ax1.set_title('Pressure Traces')
        ax1.set_ylabel('Pressure (mmHg)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot volume traces
        ax2.plot(self.ref_v, 'b-', label='Reference')
        ax2.plot(reconstructed_volume, 'r-', label='Reconstructed')
        ax2.set_title('Volume Traces')
        ax2.set_ylabel('Volume (mL)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot P-V loops
        ax3.plot(self.ref_v, self.ref_p, 'b-', label='Reference')
        ax3.set_title('Reference P-V Loop')
        ax3.set_xlabel('Volume (mL)')
        ax3.set_ylabel('Pressure (mmHg)')
        ax3.grid(True)
        
        ax4.plot(reconstructed_volume, test_pressure, 'r-', label='Reconstructed')
        ax4.set_title('Reconstructed P-V Loop')
        ax4.set_xlabel('Volume (mL)')
        ax4.set_ylabel('Pressure (mmHg)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate example data
    t = np.linspace(0, 1, 100)
    t_interp = np.linspace(0, 1, 1000)
    reference_df = pd.DataFrame({
        't': t_interp
    })
    prs_wavefrms = pd.read_csv('Pressure.csv', names=['t', 'pressure'])
    vol_wavefrms = pd.read_csv('Volume.csv', names=['t', 'volume'])

    prs_wavefrms.apply(lambda x: np.round(x, 3))
    vol_wavefrms.apply(lambda x: np.round(x, 3))
    from scipy.interpolate import pchip_interpolate

    ref_volume = pchip_interpolate(vol_wavefrms['t'], vol_wavefrms['volume'], t_interp)
    ref_pressure = pchip_interpolate(prs_wavefrms['t'], prs_wavefrms['pressure'], t_interp)

    reference_df['pressure'] = ref_pressure
    reference_df['volume'] = ref_volume
    
    rescaler = lambda x, l, u: (u-l) * ((x - min(x)) / (max(x) - min(x))) + l 
    
    # Create new pressure trace (slightly different from reference) 
    from nearpy import read_mat
    from scipy.signal import savgol_filter

    matdata = read_mat('C:/Users/Aakash/Desktop/RHC-Study/rv_sub1.mat', legacy=True)
    new_pressure = np.squeeze(matdata['RV_Templ'])
    new_pressure = savgol_filter(new_pressure, window_length=25, polyorder=2)
    new_pressure = rescaler(new_pressure, 0, 120)
    
    # Reconstruct volume
    reconstructor = RobustPVReconstructor(ref_pressure, ref_volume)
    reconstructed_volume = reconstructor.reconstruct_volume(new_pressure)
    
    # Plot results
    reconstructed_df = {
        't': t_interp, 
        'pressure': new_pressure, 
        'volume': reconstructed_volume
    }
    
    plot_waveforms_wiggers_style(reference_df)
    plot_waveforms_wiggers_style(reconstructed_df)

#%%
from dtw import dtw 
alignment = dtw(x=ref_pressure, y=new_pressure, keep_internals=True)
idx1, idx2 = alignment.index1, alignment.index2

epsilon = 0.005 # numerical stability

resistance = ref_pressure/(np.diff(ref_volume, prepend=ref_volume[0]) + epsilon) 

mapping = {test_idx: ref_idx for test_idx, ref_idx in zip(idx2, idx1)}
    
# Initialize the warped vector with NaNs
warped_vector = np.full(len(resistance), np.nan)

# Apply the warping
for test_idx, ref_idx in mapping.items():
    if test_idx < len(resistance):
        # If multiple test indices map to the same reference index, we could use different strategies:
        # Here we're just overwriting, but could also average, take first/last, etc.
        warped_vector[ref_idx] = resistance[test_idx]

# Handle any NaNs by forward-filling or interpolation
# First, find all non-NaN indices
valid_indices = np.where(~np.isnan(warped_vector))[0]

if len(valid_indices) > 0:
    # For each NaN, find the nearest valid value
    for i in range(len(warped_vector)):
        if np.isnan(warped_vector[i]):
            # Find the nearest valid index
            nearest_idx = valid_indices[np.abs(valid_indices - i).argmin()]
            warped_vector[i] = warped_vector[nearest_idx]

# Warped resistance   
plt.plot(resistance)
plt.plot(warped_vector)
# plt.plot(ref_pressure)
plt.show()
nomean = lambda x: x - np.mean(x)
# Get warped volume
new_volume = np.cumsum(nomean((new_pressure)/(warped_vector+epsilon)))
new_volume = rescaler(new_volume, np.min(ref_volume), np.max(ref_volume))
plt.plot(new_pressure)
plt.plot(new_volume)
plt.show()
plt.plot(new_pressure, new_volume)
plt.plot(ref_pressure, ref_volume)