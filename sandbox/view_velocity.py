import matplotlib.pyplot as plt
import numpy as np
from utils import ax_default, get_cycle

data = np.loadtxt(r'..\data\spt100\ui_dataset5.csv', delimiter=',', skiprows=1)
pressure = data[:, 4]
pos = data[:, 5]
vel = data[:, 6]
Np = 3

pos = pos.reshape((Np, -1))
pressure = pressure.reshape((Np, -1))
vel = vel.reshape((Np, -1))

fig, ax = plt.subplots()

colors = get_cycle("tab10", N=6)
ax.set_prop_cycle(color=colors.by_key()['color'], marker=['o', '*', '^', 's', 'D', 'X'])
for idx in range(Np):
    ax.plot(pos[idx, :]*1000, vel[idx, :], label=f'{pressure[idx, 0]:.2E} torr', markersize=5)

ax_default(ax, 'Axial distance from anode (mm)', 'Ion axial velocity ($m/s$)')
plt.show()
