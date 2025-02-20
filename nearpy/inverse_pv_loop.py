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
    
    # Create new pressure trace (slightly different from reference)
    # new_pressure = 110 * np.sin(2*np.pi*t)**2 + 15
    new_pressure = ref_pressure
    
    # Reconstruct volume
    reconstructor = PVLoopReconstructor(ref_pressure, ref_volume)
    reconstructed_volume = reconstructor.reconstruct_volume(new_pressure)
    
    # Plot results
    reconstructed_df = {
        't': t_interp, 
        'pressure': ref_pressure, 
        'volume': reconstructed_volume
    }
    
    plot_waveforms_wiggers_style(reference_df)
    plot_waveforms_wiggers_style(reconstructed_df)