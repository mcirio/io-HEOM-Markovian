import os
import sys

SCRIPT_DIR = os.path.dirname( __file__ )
WORKING_DIR = os.getcwd()
FILE_NAME = os.path.basename(__file__).split('.')[0]
relative_path_file = f"\\data\\{FILE_NAME}.dat"
MODULE_DIR = "\\".join(SCRIPT_DIR.split('\\')[:-1])
DATA_FILE = MODULE_DIR + relative_path_file
sys.path.append(MODULE_DIR)

import numpy as np
# from functools import partial
from qutip import *

from functions.saving_functions import save_dict
from functions.analytical_func import compute_system_full_analytical, compute_bath_total_analytical
from functions.utility_func import adjust
from functions.math_func import fourier
# Units
w = 2. * np.pi # frequency
ll = 1. # length
# Parameters
dimS = 2 # system dimension
ws = w # system frequency
c = ll * w # speed of light
Gamma = .4 * w # Markovian decay rate
T = 15 / w # max time
g_p = np.sqrt(Gamma * c / (2 * np.pi)) # system-bath coupling strenght
P = 10 / ll # max momentum
N_w = 1000 # number of points in frequency-space
N_T = 5000 # number of points in time
N_T_out = 15 # number of output times
N_X_out = 200 # number of output points in space
pk_list = list(np.linspace(-P,P,N_w)) # momentum discretization
Delta_p = pk_list[1] - pk_list[0] # momentum Delta
gk = g_p * np.sqrt(Delta_p) # renormalized system-bath coupling strenght
gp_list  = [g_p for _ in pk_list] # system-bath coupling in momentum
# gk_list  = [gk for pk in pk_list] # system-bath coupling in momentum
t_list = np.linspace(0,T,N_T) # dynamical times
dt = t_list[1] - t_list[0] # dt
# t_out_list = [t_list[0],t_list[-1]]
t_out_list = np.linspace(0,T,N_T_out) # out times
t_out_list = adjust(t_out_list,t_list) # out times adjusted to dynamical times
p_in = P / 10. # momentum of initial wave-packet
sigma_in = 1 / 2.   * P / 10. # standard deviation of the initial wave-packet
detuning = c * p_in / ws # intuitive detuning value between the initial wave-packet and the system
x_in = -4.5 * ll # space of initial wave-packet
Nx = N_w # number of space discretization
x_list = np.linspace(-2*abs(x_in), 2*abs(x_in), Nx) # space discretization
c1_in = 0
bath_pos_in = 1

# initial condition dict
initial_state_dic = {}
initial_state_dic['mu'] = p_in
initial_state_dic['sigma'] = sigma_in
initial_state_dic['x_in'] = x_in
initial_state_dic['c1_in'] = c1_in
initial_state_dic['bath_pos_in'] = bath_pos_in
# initial_state_dic['bath_neg_in'] = bath_neg_in

print('Computing Analytical System Dynamics')
sz_an_input_list = compute_system_full_analytical(initial_state_dic,ws,Gamma,t_list,c,pk_list,gp_list)

print('Computing Analytical Bath Dynamics')
bath_analytical = compute_bath_total_analytical(initial_state_dic,ws,Gamma,t_out_list,c,pk_list,P)

# divide positive and energy solutions
ck_an_pos_time_list = [state[:N_w] for state in bath_analytical]
ck_an_neg_time_list = [state[N_w:] for state in bath_analytical]

print('Computing the Fourier transform')
cx_an_pos_in_out_list = [fourier(ck_an_pos_time,pk_list,x_list) for ck_an_pos_time in ck_an_pos_time_list]
cx_an_neg_in_out_list = [fourier(ck_an_neg_time,pk_list,x_list) for ck_an_neg_time in ck_an_neg_time_list]

print('Saving')
data_dict = {}
data_dict['sz_an_input_list'] = sz_an_input_list
data_dict['ck_an_pos_time_list'] = ck_an_pos_time_list
data_dict['ck_an_neg_time_list'] = ck_an_neg_time_list
data_dict['t_out_list'] = t_out_list
data_dict['pk_list'] = pk_list
data_dict['x_list'] = x_list
data_dict['c'] = c
data_dict['Delta_p'] = Delta_p
data_dict['t_list'] = t_list
data_dict['x_in'] = x_in
data_dict['p_in'] = p_in
data_dict['cx_an_pos_in_out_list'] = cx_an_pos_in_out_list
data_dict['cx_an_neg_in_out_list'] = cx_an_neg_in_out_list
save_dict(data_dict,DATA_FILE)

print('Finished!')
