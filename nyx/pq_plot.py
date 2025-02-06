import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scienceplots

MU = 398600.4418    # Standard gravitational parameter of Earth
R_EARTH = 6378.137  # Radius of Earth in km
J2 = 1.08262668e-3  # J2 coefficient of Earth

# Import parquet file
df = pd.read_parquet('test.parquet')
t = df['Epoch (UTC)']
x = np.array(df['x (km)'])[:]
y = np.array(df['y (km)'])[:]
z = np.array(df['z (km)'])[:]
vx = np.array(df['vx (km/s)'])[:]
vy = np.array(df['vy (km/s)'])[:]
vz = np.array(df['vz (km/s)'])[:]
E = 0.5 * (vx**2 + vy**2 + vz**2) - MU / np.sqrt(x**2 + y**2 + z**2)

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
#    ax.plot(x, y, z, label='GL4', color='blue')
#    ax.plot(x_rk4, y_rk4, z_rk4, label='RK4', color='red', linestyle='--')
#    ax.autoscale(tight=True)
#    ax.set(**pparam)
#    ax.legend()
#    plt.show()
#    #fig.savefig('plot.png', dpi=600, bbox_inches='tight')

# Plot Energy
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(E, label='RK89', color='blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J/kg)')
    ax.legend()
    fig.savefig('plot_energy.png', dpi=600, bbox_inches='tight')
