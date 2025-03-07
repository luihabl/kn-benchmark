import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, epsilon_0 as eps, m_e
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

L = 6.7e-2
dx = L / 128
nx = 129
n0 = 2.56e14
l = 6.7e-2
ppc = 512
pweight = n0 * l / (ppc * (nx - 1))
m_i = 6.67e-27

def density(pos, nx):

    j          = np.floor(pos/dx).astype(int)
    jp1        = j+1
    weight_j   = ( jp1*dx - pos  )/dx
    weight_jp1 = ( pos    - j*dx )/dx
    n  = np.bincount(j,   weights=weight_j,   minlength=nx);
    n += np.bincount(jp1, weights=weight_jp1, minlength=nx);
    n *= n0 * l / pos.shape[0] / dx
    n[0] *= 2
    n[-1] *= 2
    return n

def count2dens(count, nx):
    return count * pweight / dx

def calc_phi(rho):
    diags = np.array([-1,0,1])
    k = np.ones(nx)
    vals  = np.vstack((k,-2*k,k))
    Lmtx = sp.spdiags(vals, diags, nx-2, nx-2)
    Lmtx = sp.lil_matrix(Lmtx)
    Lmtx /= dx**2
    Lmtx = sp.csr_matrix(Lmtx)

    res = spsolve(Lmtx, -rho[1:-1]/eps)

    return np.concatenate(([0.0], res, [0.0]))

def get_data(path):
    # pos_e = np.genfromtxt(f"{path}/pos_e.txt")
    # v_e = np.genfromtxt(f"{path}/v_e.txt", delimiter=',')
    # field_e = np.genfromtxt(f"{path}/field_e.txt")
    # field_i = np.genfromtxt(f"{path}/field_i.txt")
    # pos_i = np.genfromtxt(f"{path}/pos_i.txt")
    # v_i = np.genfromtxt(f"{path}/v_i.txt", delimiter=',')
    de = np.genfromtxt(f"{path}/density_e.txt")
    di = np.genfromtxt(f"{path}/density_i.txt")
    # rho = np.genfromtxt(f"{path}/rho.txt")

    # phi = np.genfromtxt(f"{path}/phi.txt")
    # ef = np.genfromtxt(f"{path}/efield.txt")


    # density_i = density(pos_i, nx)
    # density_e = density(pos_e, nx)
    # rho_local = e * (density_i - density_e)

    return (count2dens(di, nx), count2dens(de, nx))

d1 = get_data("build_release")
d2 = get_data("build_release/res1")

data_a = np.genfromtxt("data/Benchmark_A.csv", delimiter=' ')
x = np.linspace(0, L, d1[0].size)

# ke_e = 0.5 * m_e * (v_e[:, 0]**2 + v_e[:, 1]**2 + v_e[0, 2]**2) / e
# print(np.mean(ke_e))


# plt.hist(ke_e, bins=1000)

# ke_i = 0.5 * m_i * (v_i[:, 0]**2 + v_i[:, 1]**2 + v_i[0, 2]**2) / e
# print(np.mean(ke_i))

# plt.show()

# plt.hist(ke_i, bins=1000)

xb = data_a[:, 0]
density_e_b = data_a[:, 1]
density_i_b = data_a[:, 4]
plt.plot(xb, density_e_b, ls='--')
plt.plot(xb, density_i_b, ls='--')

plt.plot(x, d1[0])
plt.plot(x, d1[1])

plt.plot(x, d2[0])
plt.plot(x, d2[1])

# plt.plot(rho_local)
# plt.plot(rho)

# x = np.linspace(0, L, 129)

# plt.scatter(pos_e, field_e)
# plt.plot(x, ef, c='r')

# ke_e =



# plt.plot(-np.gradient(calc_phi(rho_local), dx))
# plt.plot(phi)
# plt.plot(calc_phi(rho))


# plt.plot(rho)

# phi_code = np.genfromtxt(f"{path}/phi.txt")
# efield_code = np.genfromtxt(f"{path}/efield.txt")
# print(out)



# dxv = np.ones(phi_code.size) * dx
# efield = -np.gradient(phi_code, dx, edge_order=1)


# plt.plot(efield_code)
# plt.plot(efield)

# n0 = 2.56e14
# rho = n0 * e
# L = 6.7e-2
# x = np.linspace(0, L, 129)
# dx = L / 128

# c = rho * L / (2 * eps)
# phi = c * x - rho * x**2 / (2 * eps)

# plt.plot(np.gradient(phi, dx))

plt.show()

plt.plot(d1[0] - d2[0])
plt.plot(d1[1] - d2[1])
plt.show()