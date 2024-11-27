import os
import sys

SCRIPT_DIR = os.path.dirname( __file__ )
WORKING_DIR = os.getcwd()
FILE_NAME = os.path.basename(__file__).split('.')[0][:-6]
relative_path_file = f"\\data\\{FILE_NAME}.dat"

MODULE_DIR = "\\".join(SCRIPT_DIR.split('\\')[:-1])
sys.path.append(MODULE_DIR)
DATA_FILE = MODULE_DIR + relative_path_file
PLOTS_DIR = SCRIPT_DIR + '\\plot_files'
PLOTS_FILE = PLOTS_DIR + f"\\{FILE_NAME}.pdf"

import matplotlib.pyplot as plt
import numpy as np
from functions.saving_functions import load_dict
from functions.plot_func import my_ridgeline
# Load data
print('Loading HEOM data')
data_dict = load_dict(DATA_FILE,vb=False)
ws = data_dict['ws'] 
t_list = data_dict['t_list'] 
Gamma = data_dict['Gamma'] 
pk_list = data_dict['pk_list'] 
sigma_in = data_dict['sigma_in']
sz_list = data_dict['sz_list']
c = data_dict['c']
x_out_list = data_dict['x_out_list']
obs_in = data_dict['obs_in'] 
obs_out = data_dict['obs_out'] 
obs_out_list = data_dict['obs_out_list']
t_out_list = data_dict['t_out_list']

print('Loading analytical data')
DATA_FILE = MODULE_DIR +  f"\\data\\main_analytical.dat"
data_dict = load_dict(DATA_FILE,vb=False)
sz_an_input_list = data_dict['sz_an_input_list']
ck_an_pos_time_list = data_dict['ck_an_pos_time_list'] 
ck_an_neg_time_list = data_dict['ck_an_neg_time_list'] 
t_out_list = data_dict['t_out_list']
x_list = data_dict['x_list'] 
x_in = data_dict['x_in']
p_in = data_dict['p_in']
cx_an_pos_in_out_list = data_dict['cx_an_pos_in_out_list'] 
cx_an_neg_in_out_list = data_dict['cx_an_neg_in_out_list'] 
Delta_p = data_dict['Delta_p']

ck_an_pos_in_list = ck_an_pos_time_list[0]
ck_an_pos_out_list = ck_an_pos_time_list[-1]
ck_an_neg_in_list = ck_an_neg_time_list[0]
ck_an_neg_out_list = ck_an_neg_time_list[-1]

cx_an_pos_in_list = cx_an_pos_in_out_list[0]
cx_an_pos_out_list = cx_an_pos_in_out_list[-1]
cx_an_neg_in_list = cx_an_neg_in_out_list[0]
cx_an_neg_out_list = cx_an_neg_in_out_list[-1]

t_bath = x_in / c
Delta_x = x_out_list[1] - x_out_list[0]
DeltaxFourier = x_list[1] - x_list[0]
dt = t_list[1] - t_list[0]

# Rescaling lists
t_list_rescaled = [t / abs(t_bath) for t in t_list]
x_list_rescaled = [x / abs(x_in) for x in x_list]
x_out_list_rescaled = [x / abs(x_in) for x in x_out_list]
pk_list_rescaled = [pk / abs(p_in) for pk in pk_list]

# Organize observables to be plotted 
data_analytical_list = []
data_analytical = len(cx_an_pos_in_out_list)
for cx_p,cx_m in zip(cx_an_pos_in_out_list,cx_an_neg_in_out_list):
    data_analytical = [(abs(x+y))**2  for x,y in zip(cx_p,cx_m)]
    data_analytical_list.append(data_analytical)
n_analytical_data = len(data_analytical_list)

data_HEOM_list = obs_out_list

# Plot parameters
labelsize = 20
linewidth_analytical = 6
label_analytical = 'analytical'

linewidth_HEOM = 2
color_HEOM = 'lightgrey'
linestyle_HEOM = 'solid'
label_HEOM = 'io-HEOM'

legend_fontsize = 17
fontsize = 30
labelsize = 25
x_label = r'Space$\left[|x_\text{in}|\right]$'
y_label = r'Time [$|x_\text{in}|/c$]'
markerscale = 2
linewidth_legend = '4'

# Plots.
print('Plotting environmental dynamics')
fig, ax = plt.subplots(1, 1,figsize=(14,8))

#plot analytical data
minim, maxim, eps, offset_constant = my_ridgeline(x_list_rescaled,data_analytical_list,ax,label=label_analytical,linewidth=linewidth_analytical)

# plot HEOM data
for n_HEOM,data_HEOM in enumerate(data_HEOM_list):
    # Offset the data and also make them a density and including the factor 2 corresponding to the square root definition in the analytical part.
    offset_data_HEOM = [x / Delta_x + n_HEOM * offset_constant for x in data_HEOM]
    if n_HEOM == 0:
        ax.plot(x_out_list_rescaled,offset_data_HEOM,linewidth=linewidth_HEOM,label=label_HEOM,zorder=n_analytical_data-n_HEOM,color=color_HEOM,linestyle=linestyle_HEOM)
    else:
        ax.plot(x_out_list_rescaled,offset_data_HEOM,linewidth=linewidth_HEOM,zorder=n_analytical_data-n_HEOM,color=color_HEOM,linestyle=linestyle_HEOM)
    
# Setting up the figure

ax.set_ylim(minim-eps, maxim)
ax.set_xlim(x_list_rescaled[0], x_list_rescaled[-1])
fig.set_tight_layout(True)

ax.tick_params(axis='both',which='major',labelsize=labelsize)
ax.set_xlabel(x_label,fontsize=fontsize)

leg = ax.legend(loc='upper right',fontsize=legend_fontsize,ncol=1,markerscale=markerscale)


# The y-axis plots the occupation. However, we want to highlight the units of the offset instead, i.e., time.
# a single offset corresponds to a time t_out_list[1].
# x*offset correspond to a time x*t_out_list[1].
# When x = z = abs(x_in) / (t_out_list[1]*c), z*offset = abs(x_in) / c, which is the chosen unit of time.
z = abs(x_in) / (t_out_list[1]*c)

# ax2 = ax.twinx()
ax.set_xticks([-1,0,1],[-1,0,1])
ax.set_yticks([0,z*offset_constant,2*z*offset_constant,3*z*offset_constant],[0,1,2,3])
ax.set_ylabel(y_label,fontsize=fontsize)

for line in leg.get_lines():
    line.set_linewidth(linewidth_legend)


plt.show()
# Saving.
fig.savefig(PLOTS_FILE,format='pdf')


print('Plotting system dynamics')
linewidth_system = 4
x_label = r'Time [$|x_\text{in}|/c$]'
y_label = r'$\langle \sigma_+\sigma_-\rangle$'

fig, ax = plt.subplots(1, 1,figsize=(14,8))

ax.plot(t_list_rescaled,[abs(x)**2 for x in sz_an_input_list],linestyle='solid',color='b',linewidth=linewidth_system,label=label_analytical)
ax.plot(t_list_rescaled,[x for x in sz_list],linestyle='dashed',color='r',linewidth=linewidth_system,label=label_HEOM)
ax.tick_params(axis='both',which='major',labelsize=labelsize)
ax.set_xlabel(x_label,fontsize=fontsize)
ax.set_ylabel(y_label,fontsize=fontsize)
ax.set_xticks([-0,1,2,3],[0,1,2,3])
leg = ax.legend(loc='upper right',fontsize=legend_fontsize,ncol=1,markerscale=markerscale)
fig.set_tight_layout(True)

plt.show()

print('Check of the value of the parameters as reported in the article.')
print(ws-4.5*c/abs(x_in) == 0)
print(Gamma-0.4*ws == 0)
print(sigma_in - p_in / 2. == 0)
print(p_in - ws/c == 0)