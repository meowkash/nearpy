import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

prs_wavefrms = pd.read_csv('Pressure.csv', names=['t', 'pressure'])
vol_wavefrms = pd.read_csv('Volume.csv', names=['t', 'volume'])

prs_wavefrms.apply(lambda x: np.round(x, 3))
vol_wavefrms.apply(lambda x: np.round(x, 3))

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
    
t_interp = np.linspace(0, 1, 1000)
df = pd.DataFrame({
    't': t_interp
})

from scipy.interpolate import pchip_interpolate

df['volume'] = pchip_interpolate(vol_wavefrms['t'], vol_wavefrms['volume'], t_interp)

df['pressure'] = pchip_interpolate(prs_wavefrms['t'], prs_wavefrms['pressure'], t_interp)

plot_waveforms_wiggers_style(df)