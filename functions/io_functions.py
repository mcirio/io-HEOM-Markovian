import numpy as np
from qutip import Qobj, expect
from scipy.special import erfi
from scipy.linalg import expm

# General
def compute_dyn(L0,Ltime_list,rho0,t_list):
    # evolve a linear differential equation discretized in time with infinitesimal propagator
    # L0 (time-independent) and Ltime_list (time-dependent)
    dyn_list = [rho0]
    dt = t_list[1] - t_list[0] 
    for t_index in (np.arange(1,len(t_list))):
        t = t_list[t_index]
        vec_t= dyn_list[-1]
        L = L0
        for Ltime in Ltime_list:
            L_op = Ltime[0]
            ft = Ltime[1](t)
            L = L + ft * L_op
        res = vec_t + dt * L @ vec_t       
        dyn_list.append(res)
    return dyn_list
def compute_output_parameters(time_var,t_out,Gamma,Deltax_out,c,t_list):
    value = None
    n = None
    t_var = None
    if (time_var <= t_out) & (time_var >= 0):
        value = 1j * np.sqrt(Gamma * abs(Deltax_out) / c)
        if (time_var == t_out):
            value = 0
        if (time_var == 0):
            value = value / 2.
        n = np.argmin([abs(t-time_var) for t in t_list])
        t_var = t_list[n]
    return t_var, value


def compute_dyn_kick(L0,Ltime_list,rho0,t_list,cross_dict):
    x_out = cross_dict['x_out']
    t_out = cross_dict['t_out']
    c = cross_dict['c']
    Q_out_m_full = cross_dict['Q_out_m_full']
    Q_out_p_full = cross_dict['Q_out_p_full']
    Gamma = cross_dict['Gamma']
    Deltax_out = cross_dict['Deltax_out']

    time_m = t_out - x_out / c
    t_m, value_m = compute_output_parameters(time_m,t_out,Gamma,Deltax_out,c,t_list)
    time_p = t_out + x_out / c
    t_p, value_p = compute_output_parameters(time_p,t_out,Gamma,Deltax_out,c,t_list)

    dt = t_list[1] - t_list[0] 
    dyn_list = [rho0]
    for t in t_list[:-1]:
        vec_t = dyn_list[-1]
        
        # if the time satisfies one of the two deltas, there is a kick. 
        if t == t_p:
            Q_p = value_p * (Q_out_p_full + Q_out_m_full)
            res =  expm(Q_p) @ (vec_t)
        if t == t_m:
            Q_m = value_m * (Q_out_p_full + Q_out_m_full)
            res = expm(Q_m) @ (vec_t)
        # for other times, infinitesimal evolution.
        if (t != t_p) & (t != t_m):
            L = L0
            for Ltime in Ltime_list:
                L_op = Ltime[0]
                ft = Ltime[1](t)
                L = L + ft * L_op
            res = vec_t + dt * L @ vec_t       

        dyn_list.append(res)
    return dyn_list

# Input-HEOM

def create_rho_in(rho_vec):
    # embeds the initial, vectorized state in the input-HEOM space
    dim = len(rho_vec)
    zero = np.zeros(dim)
    res = np.concatenate((rho_vec, zero,zero,zero), axis=0)
    return res

def basic_encoding_in(n_1,n_2):
    # two input indexes to a linear index
    if n_1 == 0:
        if n_2 == 0:
            return 0
        if n_2 ==1:
            return 2
    if n_1 == 1:
        if n_2 == 0:
            return 1
        if n_2 ==1:
            return 3
        
def encoding_in(n_1,n_2,dimS):
    # encoding of the input: two input indexes to a linear one, taking into account system dimension
    # the tensor structure chosen here is: system -> output
    n = basic_encoding_in(n_1,n_2)
    min_in = dimS**2 * n
    max_in = dimS**2 * (n+1)
    return [min_in,max_in]

def compute_in(obs,dyn_list,dimS,t_list):
    # compute the expectation of a system observable given the Input-HEOM solution
    sz_list = []
    for t,dyn in zip(t_list,dyn_list):
        n = encoding_in(0,0,dimS)
        n_min = n[0]
        n_max = n[1]
        rec00 = dyn[n_min:n_max]
        
        n = encoding_in(1,1,dimS)
        n_min = n[0]
        n_max = n[1]
        rec11 = dyn[n_min:n_max]
        
        rec00 = Qobj(rec00.reshape(dimS,dimS))
        rec11 = Qobj(rec11.reshape(dimS,dimS))

        sz_list.append((expect(obs,rec00) - expect(obs,rec11)) ) 
    return sz_list

def Omega_in_pm(Gamma,c,sigma_in,p_in,x_in,pm,t):
    # input time-dependent "frequency"
    tau = 1 / ( 2 * sigma_in )
    m1 = 2**(-1) * (2 * np.pi)**(-0.25) * tau**(-0.5)
    m2 = 1j * np.sqrt(Gamma * c)

    xt = x_in + c * t
    m3 = np.exp(-xt**2 / (4*tau**2) - 1j * pm * p_in * xt)
    m4 = (1 - 1j * pm * erfi(xt/(2*tau) + 1j * pm * p_in * tau))
    res1 = m1 * m2 * m3 * m4

    xmt = x_in - c * t
    m3 = np.exp(-xmt**2 / (4*tau**2) - 1j * pm * p_in * xmt)
    m4 = (1 + 1j * pm * erfi(xmt/(2*tau) + 1j * pm * p_in * tau))
    res2 = m1 * m2 * m3 * m4

    return res1 + res2

# (Input Output)-HEOM

def create_rho_in_out(rho_vec):
    # embeds the initial, vectorized state in the IO-HEOM space
    dim = len(rho_vec)
    zero = np.zeros(dim)
    res = np.concatenate((rho_vec, zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero), axis=0)
    return res
def encoding_in_out(n_in_1,n_in_2,n_out_1,n_out_2,dimS):
    # encoding of the input-output: two input indexes and two output indexes to a linear one, taking into account system dimension
    # the tensor structure chosen here is: system -> output -> input
    n_in = basic_encoding_in(n_in_1,n_in_2)
    n_out = basic_encoding_in(n_out_1,n_out_2)
    min_in = 4 * dimS**2 * n_in + n_out * dimS**2 
    max_in = min_in + dimS**2 
    return [min_in,max_in]

def cross(c,Deltax,sigma_in,p_in,x_in,x_out,t_out):
    # input time-dependent "frequency"
    tau = 1 / ( 2 * sigma_in )
    m1 = np.sqrt(Deltax) * 2**(-1) * (2 * np.pi)**(-0.25) * tau**(-0.5)

    Dx = x_out - x_in

    xmt = Dx - c * t_out
    m3 = np.exp(-xmt**2 / (4*tau**2) - 1j * p_in * xmt)
    m4 = (1 - 1j * erfi(xmt/(2*tau) + 1j * p_in * tau))
    res2 = m1 * m3 * m4

    xt = Dx + c * t_out
    m3 = np.exp(-xt**2 / (4*tau**2) - 1j * p_in * xt)
    m4 = (1 + 1j * erfi(xt/(2*tau) + 1j * p_in * tau))
    res1 = m1 * m3 * m4

    return res1 + res2

def compute_in_out(c,obs_id,dyn_list,dimS,cross_dict):
    sigma_in = cross_dict['sigma_in']
    p_in = cross_dict['p_in']
    x_in = cross_dict['x_in']
    x_out = cross_dict['x_out']
    t_out = cross_dict['t_out']
    Deltax_out  = cross_dict['Deltax_out']

    ft = cross(c,Deltax_out,sigma_in,p_in,x_in,x_out,t_out)

    obs_res_list = []
    for n,dyn in enumerate([dyn_list[0],dyn_list[-1]]):

        n = encoding_in_out(0,0,0,0,dimS)
        n_min = n[0]
        n_max = n[1]
        rec0000 = dyn[n_min:n_max]
        
        n = encoding_in_out(1,1,1,1,dimS)
        n_min = n[0]
        n_max = n[1]
        rec1111 = dyn[n_min:n_max]

        n = encoding_in_out(0,0,1,1,dimS)
        n_min = n[0]
        n_max = n[1]
        rec0011 = dyn[n_min:n_max]

        n = encoding_in_out(1,0,0,1,dimS)
        n_min = n[0]
        n_max = n[1]
        rec1001 = dyn[n_min:n_max]

        n = encoding_in_out(0,1,1,0,dimS)
        n_min = n[0]
        n_max = n[1]
        rec0110 = dyn[n_min:n_max]
        
        rec0000 = Qobj(rec0000.reshape(dimS,dimS))
        rec1111 = Qobj(rec1111.reshape(dimS,dimS))
        rec0011 = Qobj(rec0011.reshape(dimS,dimS))
        rec1001 = Qobj(rec1001.reshape(dimS,dimS))
        rec0110 = Qobj(rec0110.reshape(dimS,dimS))

        obs_res = abs(ft)**2 * expect(obs_id,rec0000) 
        obs_res += expect(obs_id,rec1111)
        obs_res += - expect(obs_id,rec0011)
        obs_res += - ft * expect(obs_id,rec1001)
        obs_res += - np.conj(ft) * expect(obs_id,rec0110)

        obs_res_list.append(obs_res) 
    return obs_res_list

