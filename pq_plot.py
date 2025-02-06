import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scienceplots

MU = 398600.4418    # Standard gravitational parameter of Earth
R_EARTH = 6378.137  # Radius of Earth in km
J2 = 1.08262668e-3  # J2 coefficient of Earth

# Import parquet file
df = pd.read_parquet('data.parquet')

# Prepare Data to Plot
x_gl4 = np.array(df['x_gl4'])[:]
y_gl4 = np.array(df['y_gl4'])[:]
z_gl4 = np.array(df['z_gl4'])[:]
vx_gl4 = np.array(df['vx_gl4'])[:]
vy_gl4 = np.array(df['vy_gl4'])[:]
vz_gl4 = np.array(df['vz_gl4'])[:]
E_gl4 = 0.5 * (vx_gl4**2 + vy_gl4**2 + vz_gl4**2) - MU / np.sqrt(x_gl4**2 + y_gl4**2 + z_gl4**2)

x_rk4 = df['x_rk4']
y_rk4 = df['y_rk4']
z_rk4 = df['z_rk4']
vx_rk4 = df['vx_rk4']
vy_rk4 = df['vy_rk4']
vz_rk4 = df['vz_rk4']
E_rk4 = 0.5 * (vx_rk4**2 + vy_rk4**2 + vz_rk4**2) - MU / np.sqrt(x_rk4**2 + y_rk4**2 + z_rk4**2)

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$y$',
    xscale = 'linear',
    yscale = 'linear',
)

## Plot
#with plt.style.context(["science", "nature"]):
#    fig = plt.figure(dpi=600)
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot(x_gl4, y_gl4, z_gl4, label='GL4', color='blue')
#    ax.plot(x_rk4, y_rk4, z_rk4, label='RK4', color='red', linestyle='--')
#    ax.autoscale(tight=True)
#    ax.set(**pparam)
#    ax.legend()
#    plt.show()
#    #fig.savefig('plot.png', dpi=600, bbox_inches='tight')

# Plot Energy
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(E_gl4, label='GL4', color='blue')
    ax.plot(E_rk4, label='RK4', color='red', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J/kg)')
    ax.legend()
    fig.savefig('plot_energy.png', dpi=600, bbox_inches='tight')
