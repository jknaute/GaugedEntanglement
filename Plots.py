import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import statsmodels.api as sm
import pickle
from mpl_toolkits.mplot3d import Axes3D

from nice_tcks import nice_ticks



## Variation of gamma

inputf = np.loadtxt("data/s2_vs_gamma.txt")

plt.scatter(inputf[:,0],inputf[:,2])
plt.xlim(0,1)
plt.ylim(0,3)
plt.xlabel("$\gamma$")
plt.ylabel("$\\bar{s}_2$")
nice_ticks()
plt.show()

plt.scatter(inputf[:,0],inputf[:,1])
# plt.xlim(0,1)
# plt.ylim(0,3)
plt.xlabel("$\gamma$")
plt.ylabel("$\\bar{p}_2$")
nice_ticks()
plt.show()


## Dependence on R2

inputf_R2_w1 = np.loadtxt("data/s2_vs_R2_R1=1.txt")
inputf_R2_w2 = np.loadtxt("data/s2_vs_R2_R1=2.txt")
inputf_R2_w3 = np.loadtxt("data/s2_vs_R2_R1=3.txt")

plt.scatter(inputf_R2_w1[:,0],inputf_R2_w1[:,2], label="$R_1=1$")
plt.scatter(inputf_R2_w2[:,0],inputf_R2_w2[:,2], label="$R_1=2$")
plt.scatter(inputf_R2_w3[:,0],inputf_R2_w3[:,2], label="$R_1=3$")

# plt.plot(inputf_R2_w1[:,0], 1.96110964e-08+2.00014215e-08*inputf_R2_w1[:,0]) # fit below
# predictions from eigenvalue ratios below:
plt.plot(inputf_R2_w1[:,0], (inputf_R2_w1[:,0]+1)*lratio, c="C0")
plt.plot(inputf_R2_w2[:,0], (inputf_R2_w2[:,0]+2)*lratio, c="C1")
plt.plot(inputf_R2_w3[:,0], (inputf_R2_w3[:,0]+3)*lratio, c="C2")

plt.xlabel("$R_2$")
plt.ylabel("$\\bar{s}_2$")
plt.legend()
nice_ticks()
plt.show()

# linear fits:
fitres_w1 = Polynomial.fit(inputf_R2_w1[:,0],inputf_R2_w1[:,2],deg=1)
print(fitres_w1.convert().coef[1])
fitres_w2 = Polynomial.fit(inputf_R2_w2[:,0],inputf_R2_w2[:,2],deg=1)
print(fitres_w2.convert().coef[1])
fitres_w3 = Polynomial.fit(inputf_R2_w3[:,0],inputf_R2_w3[:,2],deg=1)
print(fitres_w3.convert().coef[1])

# Alternative:
linmodel = sm.OLS(inputf_R2_w1[:,2], sm.add_constant(inputf_R2_w1[:,0])) # y,x
res = linmodel.fit()
res.params

intercepts = [fitres_w1.convert().coef[0], fitres_w2.convert().coef[0], fitres_w3.convert().coef[0]]

plt.scatter(range(1,4), intercepts)


# Case alpha = delta
inputf_R2_ad = np.loadtxt("data/s2_vs_R2_R1=2_alpha=delta.txt")

plt.scatter(inputf_R2_ad[:,0],inputf_R2_ad[:,2], label="$R_1=2$")

plt.xlabel("$R_2$")
plt.ylabel("$\\bar{s}_2$")
plt.legend()
nice_ticks()
plt.show()

fitres = Polynomial.fit(inputf_R2_ad[:,0],inputf_R2_ad[:,2],deg=1)
print(fitres.convert().coef[1])

# nonperturbative regime
inputf_R2_nonpert = np.loadtxt("data/s2_vs_R2_nonpert.txt")

plt.scatter(inputf_R2_nonpert[:,0],inputf_R2_nonpert[:,2], label="$R_1=2$")
plt.xlabel("$R_2$")
plt.ylabel("$\\bar{s}_2$")
plt.legend()
nice_ticks()
plt.show()

fitres = Polynomial.fit(inputf_R2_nonpert[:,0],inputf_R2_nonpert[:,2],deg=1)
print(fitres.convert().coef[1])


## Variation of delta
inputf_delta = np.loadtxt("data/s2_vs_delta.txt")

plt.scatter(inputf_delta[:,0],inputf_delta[:,2], label="$\\alpha=1$")

plt.xlabel("$\\delta/\\alpha$")
plt.ylabel("$\\bar{s}_2$")
plt.legend()
nice_ticks()
plt.show()


## Dependence of eigenvalues on R1

# perturbative example

run Entropies_numerical_eigenvalues.py

lratio = -np.log(rho2p_dom[0,0]/rho2_dom[0])
lratio

# nonperturbative case

run Entropies_numerical_eigenvalues.py



## Parameter grid

# perturbative regime

file = open("data/s2_vs_delta_gamma.p", "rb")
grid_pert = pickle.load(file)
file.close()

g,d = np.meshgrid(grid_pert['gamma_vals'], grid_pert['delta_vals'])

fig = plt.figure()
plt.imshow(grid_pert['s2'], origin='lower', extent=[0.5,1.5,0,2])
plt.xlabel("$\\delta$")
plt.ylabel("$\\gamma$")
plt.show()

# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# x,y = np.meshgrid(grid_pert['gamma_vals'], grid_pert['delta_vals'])
# ax.scatter(x, y, grid_pert['s2'].T)
# plt.show()

fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
surf = ax.plot_surface(d, g, grid_pert['s2'].T, cmap='viridis')
ax.view_init(azim=200)
plt.xlabel("$\\delta$")
plt.ylabel("$\\gamma$")
# plt.zlabel("$s_2$")
clb = fig.colorbar(surf, shrink=0.7, aspect=10) #, label="$\\bar{s}_2$")
clb.ax.set_title("$\\bar{s}_2$")
plt.show()

plt.contour(d, g, grid_pert['s2'].T, levels=np.linspace(0,100))

































