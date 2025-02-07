import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import beaupy
from rich.console import Console

console = Console()
console.print("[bold green]Plotting Energy...[/bold green]")
console.print("Select dynamics: ")
options = ["J2", "2BD"]
dynamics = beaupy.select(options)

MU = 398600.4418    # Standard gravitational parameter of Earth
R_EARTH = 6378.137  # Radius of Earth in km
J2 = 1.08262668e-3  # J2 coefficient of Earth

def add_J2_to_energy(x, y, z, vx, vy, vz):
    r = np.sqrt(x**2 + y**2 + z**2)
    r2 = r**2
    r3 = r**3
    U_j2 = MU * J2 * R_EARTH**2 / (2 * r3) * (3 * z**2 / r2 - 1)
    return U_j2

def energy_from_orbit(model: str, dynamics: str):
    df = pl.read_parquet(f'data_{model}_{dynamics}.parquet')
    t = np.array(df['t'])[::100]
    x = np.array(df['x'])[::100]
    y = np.array(df['y'])[::100]
    z = np.array(df['z'])[::100]
    vx = np.array(df['vx'])[::100]
    vy = np.array(df['vy'])[::100]
    vz = np.array(df['vz'])[::100]
    J2_energy = add_J2_to_energy(x, y, z, vx, vy, vz) if dynamics == 'J2' else 0
    E = 0.5 * (vx**2 + vy**2 + vz**2) - MU / np.sqrt(x**2 + y**2 + z**2) + J2_energy
    return t, E

# Import parquet file
t_yoshida, E_yoshida = energy_from_orbit('yoshida', dynamics)
t_rk4, E_rk4 = energy_from_orbit('rk4', dynamics)
t_dp45, E_dp45 = energy_from_orbit('dp45', dynamics)
t_gl4, E_gl4 = energy_from_orbit('gl4', dynamics)

# Plot params
pparam = dict(
    xlabel = r'Time (s)',
    ylabel = r'Energy (J/kg)',
    xscale = 'linear',
    yscale = 'linear',
)

# Plot Energy
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(t_yoshida, E_yoshida, label='Yoshida', color='blue', alpha=0.5)
    ax.plot(t_rk4, E_rk4, label='RK4', color='red', linestyle='--', alpha=0.5)
    ax.plot(t_dp45, E_dp45, label='DP45', color='green', linestyle='-.', alpha=0.5)
    ax.plot(t_gl4, E_gl4, label='GL4', color='orange', linestyle='-', alpha=0.5)
    ax.legend()
    fig.savefig(f'plot_energy_{dynamics}.png', dpi=600, bbox_inches='tight')
