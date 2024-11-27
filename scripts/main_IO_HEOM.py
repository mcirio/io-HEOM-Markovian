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
from functools import partial
from qutip import *

from functions.saving_functions import save_dict
from functions.progressbar import progressbar
from functions.quantum_functions import Hamiltonian_single, Lindblad_single
from functions.quantum_functions import sigmaz_op,sigmay_op,sigmax_op,id_op, sigmap_op, sigmam_op
from functions.io_functions import compute_dyn, compute_dyn_kick
from functions.io_functions import  create_rho_in, compute_in, Omega_in_pm
from functions.io_functions import create_rho_in_out, compute_in_out
from functions.utility_func import adjust, partial_list

ID2 = np.array([[1,0],[0,1]])
TM = np.array([[0,0],[1,0]])
ID4 = np.kron(ID2,ID2)

# Units
w = 2. * np.pi # frequency
ll = 1. # length
# Parameters
dimS = 2 # system dimension
ws = w # system frequency
c = ll * w # speed of light
Gamma = .4 * w # Markovian decay rate
T = 15 / w # max time
# t_out = T # 
g_p = np.sqrt(Gamma * c / (2 * np.pi)) # system-bath coupling strenght
P = 10 / ll # max momentum
N_w = 1000 # number of points in frequency-space
N_T = 5000 # number of points in time
N_T_out = 15 # number of output times
N_X_out = 500 # number of output points in space #!200
# system_list = [ws] #
pk_list = list(np.linspace(-P,P,N_w)) # momentum discretization
Delta_p = pk_list[1] - pk_list[0] # momentum Delta
gk = g_p * np.sqrt(Delta_p) # renormalized system-bath coupling strenght
gk_list  = [gk for _ in pk_list] # system-bath coupling in momentum
t_list = np.linspace(0,T,N_T) # dynamical times
dt = t_list[1] - t_list[0] # dt
# t_out_list = [t_list[0],t_list[-1]]
t_out_list = np.linspace(0,T,N_T_out) # out times
t_out_list = adjust(t_out_list,t_list) # out times adjusted to dynamical times
p_in = P / 10. # momentum of initial wave-packet
sigma_in = 1 / 2.   * P / 10. # standard deviation of the initial wave-packet
detuning = c * p_in / ws # intuitive detuning value between the initial wave-packet and the system
x_in = -4.5 * ll # space of initial wave-packet
Nx = 1000 # number of space discretization
# x_list = np.linspace(-10 * ll, 10 * ll, Nx) # space discretization
x_out_list = np.linspace(-2*abs(x_in), 2*abs(x_in), N_X_out) # out-space discretization
Deltax_out = abs(x_out_list[1] - x_out_list[0]) # out space resolution

# Quantum variables
H_S = ws * sigmap_op * sigmam_op # system Hamiltonian
L0 = (Hamiltonian_single(H_S) + Gamma * Lindblad_single(sigmam_op)).full() # Markovian decay
# L0 = (Gamma * Lindblad_single(sigmam_op)).full()
rho0_qutip = basis(2,0) * basis(2,0).dag() # initial system state
rho_vec = (operator_to_vector(rho0_qutip).full()).reshape(dimS**2) # initial state vectorized
rho0 = create_rho_in(rho_vec) # embeds initial state in the input-HEOM space
obs = sigmap_op * sigmam_op # system observable

print('I-HEOM (starting)')
# superoperators on the system space
Q_in_p = -1j * ( spre(sigmap_op) - spost(sigmap_op) ).full()
Omega_in_p_t = partial(Omega_in_pm,Gamma,c,sigma_in,p_in,x_in,1)
Q_in_m = -1j * ( spre(sigmam_op) - spost(sigmam_op) ).full()
Omega_in_m_t = partial(Omega_in_pm,Gamma,c,sigma_in,p_in,x_in,-1)

# superoperators in input-HEOM space
L0_full = np.kron(np.kron(ID2,ID2),L0)
Q_in_p_full = np.kron(np.kron(ID2,TM),Q_in_p)
Q_in_m_full = np.kron(np.kron(TM,ID2),Q_in_m)

# separating time-independent (Markovian dynamics) and time-dependent (input) contributions
Ltime_list = []
Ltime_list.append([Q_in_p_full,Omega_in_p_t])
Ltime_list.append([Q_in_m_full,Omega_in_m_t])

# evolve input-HEOM
dyn_list = compute_dyn(L0_full,Ltime_list,rho0,t_list)
# compute system observable from the input-HEOM solution
sz_list = compute_in(obs,dyn_list,dimS,t_list)
print('I-HEOM (finished)')

print('IO-HEOM (starting)')
# superoperators on the system space
Q_out_p = 1j * spost(sigmap_op).full()
Q_out_m = - 1j * spre(sigmam_op).full()

# superoperators on the IO-HEOM space
L0_full = np.kron(np.kron(ID4,ID4),L0)

Q_in_p_full = np.kron(np.kron(np.kron(ID2,TM),ID4),Q_in_p)
Q_in_m_full = np.kron(np.kron(np.kron(TM,ID2),ID4),Q_in_m)

Q_out_p_full = np.kron(np.kron(ID4,np.kron(ID2,TM)),Q_out_p)
Q_out_m_full = np.kron(np.kron(ID4,np.kron(TM,ID2)),Q_out_m)

rho0 = create_rho_in_out(rho_vec) # input-output embedding
obs_id = qeye(dimS) # output observable is already encoded in the fields.
cross_dict = {} # dictionary for the cross-correlation between the output and the bath-coupling operator.
cross_dict['sigma_in'] = sigma_in
cross_dict['p_in'] = p_in
cross_dict['x_in'] = x_in
cross_dict['Deltax_out'] = Deltax_out
cross_dict['Gamma'] = Gamma
cross_dict['c'] = c
cross_dict['Q_out_m_full'] = Q_out_m_full
cross_dict['Q_out_p_full'] = Q_out_p_full
obs_out_list = []
for t_index in progressbar(np.arange(0,len(t_out_list))):
    t_out = t_out_list[t_index]

    partial_t_list = partial_list(t_list,t_out)
    cross_dict['t_out'] = t_out
    cross_dict['partial_t_list'] = partial_t_list
    obs_in = []
    obs_out = []
    for x_index in (np.arange(0,len(x_out_list))):
        x_out = x_out_list[x_index] # output position
        cross_dict['x_out'] = x_out

        # put together all the pieces
        Ltime_list = []
        Ltime_list.append([Q_in_p_full,Omega_in_p_t])
        Ltime_list.append([Q_in_m_full,Omega_in_m_t])

        # Compute the dynamics
        dyn_list = compute_dyn_kick(L0_full,Ltime_list,rho0,partial_t_list,cross_dict)
        # Reconstruct the observable
        res = compute_in_out(c,obs_id,dyn_list,dimS,cross_dict)

        # the meaningful quantities are the initial value and the final one (evaluated at t_out)
        obs_in.append(res[0])
        obs_out.append(res[-1])
    obs_out_list.append(obs_out)
print('IO-HEOM (finshed)')

print('Saving results')
data_dict = {}
data_dict['ws'] = ws
data_dict['t_list'] = t_list
data_dict['Gamma'] = Gamma
data_dict['pk_list'] = pk_list
data_dict['x_in'] = x_in
data_dict['sigma_in'] = sigma_in
data_dict['c']  = c
data_dict['Delta_p'] = Delta_p
data_dict['sz_list'] = sz_list
data_dict['x_out_list'] = x_out_list
data_dict['obs_in'] = obs_in
data_dict['obs_out'] = obs_out
data_dict['obs_out_list'] = obs_out_list
data_dict['t_out_list'] = t_out_list
save_dict(data_dict,DATA_FILE)

print('Finished!')
