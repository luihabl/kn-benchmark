import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, epsilon_0 as eps

out = np.genfromtxt("build/output.txt")
print(out)

plt.plot(out)

n0 = 2.56e14
rho = n0 * e
L = 6.7e-2
x = np.linspace(0, L, 129)
dx = L / 128

c = rho * L / (2 * eps)
phi = c * x - rho * x**2 / (2 * eps)

# plt.plot(np.gradient(phi, dx))

plt.show()