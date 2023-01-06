import matplotlib.pyplot as plt
import numpy as np
from utils import ax_default

data = np.loadtxt('..\data\jion_dataset4.csv', delimiter=',', skiprows=1)
pressure = data[:, 4]
angle = data[:, 6]
jion = data[:, 7]
Np = 8

angle = angle.reshape((Np, -1))
pressure = pressure.reshape((Np, -1))
jion = jion.reshape((Np, -1))

fig, ax = plt.subplots()
for idx in range(Np):
    ax.plot(angle[idx, :], jion[idx, :], label=f'{pressure[idx, 0]:.2E} torr')

ax.set_yscale('log')
ax_default(ax, 'Angle from thruster centerline (deg)', 'Ion current density ($mA/cm^2$)')
plt.show()
