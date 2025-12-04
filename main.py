# -*- coding: utf-8 -*-
"""
Hypersonic Boost-Glide Monte Carlo Dispersion Analysis 

Astin X. Teel | December 2025 

6-DOF-style integration with realistic dispersions
Generates 99% containment ellipse + IIP scatter 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse 
from scipy.stats import chi2

# Vehicle and Nominal Trajectory Parameters
V_bo = 6500.0          # Burnout Velocity [m/s] (Mach ~20 at 40km)
gamma_bo = -25.0       # Flight path angle as burnout [deg]
h_bo = 45000.0         # Burnout Altitude [m]
R_earth = 6371000.0    # Earth Radius [m]

# Dispersion Standard Deviations (realistic 3 sigma)
sigma_vel = 35.0       # [m/s] (thrust/Isp variation)
sigma_fpa = 0.8        # [deg] (attitude knowledge/alignment)
sigma_drag = 0.12      # +-12% Cd (hypersonic uncertainty)
sigma_mass = 0.04      # +-4% mass
sigma_wind = 25.0      # [m/s] horizontal wind (GRAM profile)

# Monte Carlo Settings
N = 5000               # Number of Runs
np.random.seed(42)

# Storage
impact_x = np.zeros(N)
impact_y = np.zeros(N)

print(f"Running {N} Monte Carlo cases...")

for i in range(N):
    # Applying Dispersions
    vel = V_bo * (1 + sigma_vel/V_bo * np.random.randn())
    fpa = np.deg2rad(gamma_bo + sigma_fpa * np.random.randn())
    Cd_factor = 1 + sigma_drag * np.random.randn()
    m_factor = 1 + sigma_mass * np.random.randn()
    wind_x = sigma_wind * np.random.randn()
    
    # Simplified analytic boost_glide (for constant L/D glide)
    
    LD = 2.5
    g0 = 9.81
    H = 7000
    
    # Range to apex then glide
    r_bo = R_earth + h_bo
    sin_gamma = np.sin(fpa)
    range_to_apex = R_earth + np.arctan2(vel * sin_gamma *r_bo,
                                         r_bo * g0 * (r_bo - vel**2/g0 *np.cos(fpa)**2))
    
    glide_range = (H * LD) * np.log(1 + vel**2 * np.cos(fpa)**2 / (g0 * H * LD))
    total_range = range_to_apex + glide_range * Cd_factor/m_factor
    
    total_range += wind_x * 60
    
    impact_x[i] = total_range * np.cos(np.deg2rad(gamma_bo)) / 111320.0 
    impact_y[i] = total_range * np.sin(np.deg2rad(gamma_bo)) / 111320.0
    
    if (i+1) % 1000 == 0:
        print(f"   {i+1} cases completed")
        
mean_x = np.mean(impact_x)
mean_y = np.mean(impact_y)
centered_x = impact_x - mean_x
centered_y = impact_y - mean_y

cov = np.cov(centered_x, centered_y)
eigvals, eigvecs = np.linalg.eigh(cov)
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

chi2_val = chi2.ppf(0.99, df = 2)
width, height =2 * np.sqrt(chi2_val * eigvals)
angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))

plt.figure(figsize = (10, 8))
plt.scatter(impact_x, impact_y, s = 4, alpha = 0.6, c = 'steelblue', label = 'Instantaneous Impact Points')
plt.scatter(mean_x, mean_y, c = 'red', s = 100, marker = 'x', label = 'Mean IIP')

ellipse = Ellipse(xy = (mean_x, mean_y), width = width, height = height, angle = angle,
                  linewidth = 2, edgecolor = 'darkred', fc = 'none', label = '99% Containment Ellipse')
plt.gca().add_patch(ellipse)

plt.xlabel('Downrange Dispersion (deg latitude)')
plt.ylabel("Crossrange Dispersion (deg longitude")
plt.title("Hypersonic Boost-Glide Monte Carlo Dispersion \n5,000 Cases | 99% Containment Ellipse")
plt.legend()
plt.grid(True, alpha = 0.3)
plt.axis('equal')
plt.tight_layout()
# plt.savefig('results/impact_ellipse.png', dpi = 300)
plt.show()


print(f"\nDone! 99% ellipse semi-axes: {width/2:.3f} x {height/2:.3f}")
