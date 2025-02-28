import pandas as pd 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 
from scipy.interpolate import pchip_interpolate

def load_wiggers(): 
    ref_path = Path.cwd() / 'wiggers_reference'
    lv_pressure = pd.read_csv(ref_path / 'LVP.csv', names=['t', 'pressure'])
    lv_volume = pd.read_csv(ref_path / 'LVV.csv', names=['t', 'volume'])

    lv_pressure.apply(lambda x: np.round(x, 3))
    lv_volume.apply(lambda x: np.round(x, 3))
    
    interp_time = np.linspace(0, 1, 1000)
    
    wiggers_df = pd.DataFrame({
        'time': interp_time,
        'lv_vol': pchip_interpolate(lv_volume['t'], lv_volume['volume'], interp_time),
        'lv_prs': pchip_interpolate(lv_pressure['t'], lv_pressure['volume'], interp_time)
    })

    return wiggers_df

def plot_wiggers_waveforms():
    wiggers_df = load_wiggers() 
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot pressure and volume waveforms
    ax1.plot(wiggers_df['time'], wiggers_df['lv_prs'], 'r-', linewidth=2, label='LV Pressure')
    ax1.set_ylabel('LV Pressure (mmHg)')
    ax1.set_ylim(-10, 140)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Wiggers Diagram')
    
    # Add second y-axis for volume
    ax1_twin = ax1.twinx()
    ax1_twin.plot(wiggers_df['time'], wiggers_df['lv_vol'], 'k-', linewidth=2, label='LV Volume')
    ax1_twin.set_ylabel('LV Volume (ml)')
    
    # Add cardiac phases
    ax1.axvspan(0, 0.065, alpha=0.2, color='gray', label='IVC')
    ax1.axvspan(0.299, 0.38, alpha=0.2, color='gray', label='IVR')
    
    # Add labels for key events
    # ax1.text(0.02, 30, 'Mitral\nValve\nClosing', fontsize=8)
    ax1.text(-0.035, 90, 'Aortic\nValve\nOpening', fontsize=12, fontweight='bold')
    # ax1.text(0.31, 60, 'Aortic\nValve\nClosing', fontsize=8)
    ax1.text(0.43, 5, 'Mitral\nValve\nOpening', fontsize=12, fontweight='bold')
    
    # Plot P-V loop
    ax2.plot(wiggers_df['lv_vol'], wiggers_df['lv_prs'], 'k-', linewidth=2)
    ax2.set_xlabel('LV Volume (ml)')
    ax2.set_ylabel('LV Pressure (mmHg)')
    ax2.set_ylim(-10, 150)
    ax2.set_xlim(0, 150)
    ax2.set_title('Pressure-Volume Loop')
    ax2.grid(True, alpha=0.3)
    
    # Add labels for key points on P-V loop
    ax2.text(wiggers_df['lv_vol'][0]-5, wiggers_df['lv_prs'][0], 'EDV', fontsize=8)
    min_vol_idx = np.argmin(wiggers_df['lv_vol'])
    ax2.text(wiggers_df['lv_vol'][min_vol_idx]-5, wiggers_df['lv_prs'][min_vol_idx], 'ESV', fontsize=8)
    
    plt.tight_layout()
    plt.show()