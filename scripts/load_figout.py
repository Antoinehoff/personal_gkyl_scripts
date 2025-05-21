import pygkyl
import matplotlib.pyplot as plt
import numpy as np
simulation = pygkyl.Simulation()

plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["font.size"] = 16
figdatadict = pygkyl.fig_tools.load_figout('fig_profiles')
n_exp_file = 'ne_exp.txt'
Te_exp_file = 'Te_exp.txt'
delimiter = ','

n_plot = figdatadict[0]
ne_sim = n_plot['curves'][0]
ne_exp = {}
ne_exp['label'] =r'$n_e^{exp}$';
data = np.loadtxt(n_exp_file, delimiter=delimiter) 
ne_exp['xdata'] = data[:,0]/simulation.normalization['xscale']+1;
ne_exp['ydata'] = data[:,1];

plt.plot(ne_sim['xdata'],ne_sim['ydata'],label=ne_sim['label'])
plt.plot(ne_exp['xdata'],ne_exp['ydata'],'ok',label=ne_exp['label'])
plt.legend()
plt.ylabel(r'1/m${^3}$')
plt.xlabel('r/a')
plt.show()

T_plot = figdatadict[2]
Te_sim = T_plot['curves'][0]
Te_exp = {}
Te_exp['label'] =r'$T_e^{exp}$';
data = np.loadtxt(Te_exp_file, delimiter=delimiter) 
Te_exp['xdata'] = data[:,0]/simulation.normalization['xscale']+1;
Te_exp['ydata'] = data[:,1];

plt.plot(Te_sim['xdata'],Te_sim['ydata'],label=Te_sim['label'])
plt.plot(Te_exp['xdata'],Te_exp['ydata'],'ok',label=Te_exp['label'])
plt.legend()
plt.ylabel('eV')
plt.xlabel('r/a')
plt.show()