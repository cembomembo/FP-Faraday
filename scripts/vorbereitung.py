import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

files = ['N-BK7.csv', 'N-SF10.csv']
csv_delimiter = ',' 

def analyze_glass_dispersion(filename):
    print(f"--- Analyzing {filename} ---")
    
    try:
        df = pd.read_csv(filename, delimiter=csv_delimiter, header=None)
        lamb = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna() # Wavelength in micrometers
        n = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()    # Refractive Index
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Model: 1/(n^2 - 1) = -A * (1/lambda^2) + B
    # y = 1/(n^2 - 1)
    # x = 1/lambda^2
    # Filter to visible range (RECOMMENDED)
    mask = (lamb >= 0.4) & (lamb <= 0.8)  # Keep only from 400nm
    lamb = lamb[mask]
    n = n[mask]
    
    y_data = 1 / (n**2 - 1)
    x_data = 1 / (lamb**2)

    # Linear Fit (y = m*x + c)
    def linear_model(x, m, c):
        return m * x + c

    popt, pcov = curve_fit(linear_model, x_data, y_data)
    m_fit, c_fit = popt
    
    # Slope magnitude A = |m|
    # Intercept B = c
    # Resonance Wavelength lambda_R = sqrt(A / B)
    
    A = abs(m_fit)
    B = c_fit
    lambda_R = np.sqrt(A / B)
    
    print(f"Linear Fit Results:")
    print(f"  Slope (m)     : {m_fit:.6f} (matches -A)")
    print(f"  Intercept (c) : {c_fit:.6f} (matches A/lambda_R^2)")
    print(f"Calculated Parameters:")
    print(f"  Parameter A   : {A:.6f} um^2")
    print(f"  Resonance λ_R : {lambda_R:.6f} um ({lambda_R*1000:.1f} nm)")
    print("-" * 30)

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot 1: Refractive Index vs Wavelength
    ax1.plot(lamb, n, 'b.', label='Data')
    ax1.set_xlabel(r'Wavelength $\lambda$ ($\mu m$)')
    ax1.set_ylabel(r'Refractive Index $n$')
    ax1.set_title(r'Refractive Index $n(\lambda)$')
    ax1.grid(True)

    # Plot 2: Linearization (Task 1)
    # Plot Data
    ax2.plot(x_data, y_data, 'rx', label='Linearized Data')
    
    # Plot Fit Line
    x_range = np.linspace(min(x_data), max(x_data), 100)
    y_fit = linear_model(x_range, m_fit, c_fit)
    ax2.plot(x_range, y_fit, 'k-', label=f'Fit: $\lambda_R={lambda_R*1000:.1f}$ nm')
    
    ax2.set_xlabel(r'$1 / \lambda^2$ ($\mu m^{-2}$)')
    ax2.set_ylabel(r'$1 / (n^2 - 1)$')
    ax2.set_title(r'Single-Oscillator Model Linearization')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    plot_filename = filename.replace('.csv', '_plot.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as: {plot_filename}")
    
    plt.show()

# Main
if __name__ == "__main__":
    import os
    for file in files:
        analyze_glass_dispersion(file)