import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scienceplots

MU = 398600.4418    # Standard gravitational parameter of Earth
R_EARTH = 6378.137  # Radius of Earth in km
J2 = 1.08262668e-3  # J2 coefficient of Earth

# Import parquet file
df = pd.read_parquet('data_yoshida.parquet')
t_yoshida = np.array(df['t_yoshida'])[:]
x_yoshida = np.array(df['x_yoshida'])[:]
y_yoshida = np.array(df['y_yoshida'])[:]
z_yoshida = np.array(df['z_yoshida'])[:]
vx_yoshida = np.array(df['vx_yoshida'])[:]
vy_yoshida = np.array(df['vy_yoshida'])[:]
vz_yoshida = np.array(df['vz_yoshida'])[:]
E_yoshida = 0.5 * (vx_yoshida**2 + vy_yoshida**2 + vz_yoshida**2) - MU / np.sqrt(x_yoshida**2 + y_yoshida**2 + z_yoshida**2)

df = pd.read_parquet('data_rk4.parquet')
t_rk4 = df['t_rk4']
x_rk4 = df['x_rk4']
y_rk4 = df['y_rk4']
z_rk4 = df['z_rk4']
vx_rk4 = df['vx_rk4']
vy_rk4 = df['vy_rk4']
vz_rk4 = df['vz_rk4']
E_rk4 = 0.5 * (vx_rk4**2 + vy_rk4**2 + vz_rk4**2) - MU / np.sqrt(x_rk4**2 + y_rk4**2 + z_rk4**2)

df = pd.read_parquet('data_dp45.parquet')
t_dp45 = df['t_dp45']
x_dp45 = df['x_dp45']
y_dp45 = df['y_dp45']
z_dp45 = df['z_dp45']
vx_dp45 = df['vx_dp45']
vy_dp45 = df['vy_dp45']
vz_dp45 = df['vz_dp45']
E_dp45 = 0.5 * (vx_dp45**2 + vy_dp45**2 + vz_dp45**2) - MU / np.sqrt(x_dp45**2 + y_dp45**2 + z_dp45**2)

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
#    ax.plot(x_yoshida, y_yoshida, z_yoshida, label='GL4', color='blue')
#    ax.plot(x_rk4, y_rk4, z_rk4, label='RK4', color='red', linestyle='--')
#    ax.autoscale(tight=True)
#    ax.set(**pparam)
#    ax.legend()
#    plt.show()
#    #fig.savefig('plot.png', dpi=600, bbox_inches='tight')

# Plot Energy
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(t_yoshida, E_yoshida, label='Yoshida', color='blue')
    ax.plot(t_rk4, E_rk4, label='RK4', color='red', linestyle='--')
    ax.plot(t_dp45, E_dp45, label='DP45', color='green', linestyle='-.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J/kg)')
    ax.legend()
    fig.savefig('plot_energy.png', dpi=600, bbox_inches='tight')
