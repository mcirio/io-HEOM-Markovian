import numpy as np
from functools import partial

from functions.progressbar import progressbar
from functions.math_func import gaussian
from functions.integral import integral


def dispersion(p:float,c:float) -> float:
    return c * np.abs(p)

# System dynamics

def compute_system_full_analytical(initial_state_dic:dict,ws:float,Gamma:float,t_list:list,c:float,pk_list:list,gp_list:list) -> list:
    p_in = initial_state_dic['mu']
    sigma_in = initial_state_dic['sigma'] 
    x_in = initial_state_dic['x_in'] 
    c1_in = initial_state_dic['c1_in'] 
    bath_pos_in = initial_state_dic['bath_pos_in']
    Delta_p = pk_list[1] - pk_list[0]

    ck_pos_in_list = [bath_pos_in * np.sqrt(gaussian(pk,p_in,sigma_in)) * np.exp(-1j * pk * x_in) * np.sqrt(Delta_p) for pk in pk_list]
    #!!!
    ck_pos_in_list = [bath_pos_in * np.sqrt(gaussian(pk,p_in,sigma_in)) * np.exp(-1j * pk * x_in)  for pk in pk_list]

    ck_neg_in_list = [0 for _ in pk_list]


    dynamics_list = []

    for t_index in progressbar(np.arange(len(t_list))):
        t = t_list[t_index]
        res = c1_in * np.exp((-1j * ws - Gamma ) * t) 
        for pk,gk,ck_pos_in,ck_neg_in in zip(pk_list,gp_list,ck_pos_in_list,ck_neg_in_list):
            w = dispersion(pk,c)
            num = np.exp(-1j * w*t) - np.exp((-1j * ws - Gamma ) * t) 
            den = 1j * ( ws - w ) + Gamma
            res += -1j * gk * ck_pos_in * num / den 

            w = - dispersion(pk,c)
            num = np.exp(-1j * w*t) - np.exp((-1j * ws - Gamma ) * t) 
            den = 1j * ( ws - w ) + Gamma
            res +=  -1j * gk * ck_neg_in * num / den
        dynamics_list.append(Delta_p * res)
    return dynamics_list


# Bath dynamics

def to_integrate_1(p:float,initial_state_dic:dict,Gamma:float,c:float,ws:float,t:float,pos_neg:str,p_bar:float):
    eps = 10**-12

    mu = initial_state_dic['mu']
    sigma = initial_state_dic['sigma'] 
    x_in = initial_state_dic['x_in'] 
    # c1_in = initial_state_dic['c1_in'] 
    bath_pos_in = initial_state_dic['bath_pos_in']

    gp = np.sqrt(Gamma * c / (2 * np.pi))

    if pos_neg == 'pos' : 
        w = dispersion(p,c)
    if pos_neg == 'neg' : 
        w = - dispersion(p,c)

    # The integral is over both positive and negative energy states.
    # However, here only positive ones are accounted for because negative energy modes are not present in the initial condition.
    # To include negative energy modes we can just add the same terms as below, after changing the initial condition and adding a minus to the energy.
    in_cond_pos_en = ( bath_pos_in * np.sqrt(gaussian(p_bar,mu,sigma)) * np.exp(-1j * p_bar * x_in) )
    w_bar = dispersion(p_bar,c)
    res_pos_1 =  -( np.exp(-1j*w_bar*t) - np.exp(-1j*w*t) ) / (1j * (w-w_bar)+eps)
    res_pos_2 = +( np.exp(-(1j*ws+Gamma)*t) - np.exp(-1j*w*t) ) / (1j * (w-ws)-Gamma)
    res_pos_3 =  gp**2  *  in_cond_pos_en / (1j * (ws-w_bar)+Gamma)
       
    return res_pos_3 * (res_pos_1+res_pos_2) 

def bath_total_analytical(p:float,initial_state_dic:dict,Gamma:float,c:float,ws:float,t:float,pos_neg:str) -> float:
    mu = initial_state_dic['mu']
    sigma = initial_state_dic['sigma'] 
    x_in = initial_state_dic['x_in'] 
    c1_in = initial_state_dic['c1_in'] 
    bath_pos_in = initial_state_dic['bath_pos_in']

    gp = np.sqrt(Gamma * c / (2 * np.pi))

    if pos_neg == 'pos' : 
        w = dispersion(p,c)
        cp_in = bath_pos_in * np.sqrt(gaussian(p,mu,sigma)) * np.exp(-1j * p * x_in) 
    if pos_neg == 'neg' : 
        w = - dispersion(p,c)
        cp_in = 0

    res = np.exp(-1j * w * t) * cp_in
    res += -1j * np.conj(gp) * c1_in * ( np.exp(-(1j*ws+Gamma) * t) - np.exp((-1j*w)*t) )/ (1j * (w-ws)-Gamma)
    
    return res

def compute_bath_total_analytical_single_time(initial_state_dic:dict,ws:float,Gamma:float,time:float,c:float,pk_list:list,P:float) -> list:
    NNP = 20
    mu = initial_state_dic['mu']
    sigma = initial_state_dic['sigma']
    Delta_p = pk_list[1] - pk_list[0]
    ck_list = []
    for p in pk_list:
        fp = partial(to_integrate_1,p,initial_state_dic,Gamma,c,ws,time,'pos')
        res = bath_total_analytical(p,initial_state_dic,Gamma,c,ws,time,'pos')
        res += integral(fp,x_i=-P,x_f=P)
        ck_list.append(res)
    for p in pk_list:
        fp = partial(to_integrate_1,p,initial_state_dic,Gamma,c,ws,time,'neg')
        res = bath_total_analytical(p,initial_state_dic,Gamma,c,ws,time,'neg')
        res += integral(fp,x_i=-P,x_f=P)
        ck_list.append( res)

    return ck_list
def compute_bath_total_analytical(initial_state_dic:dict,ws:float,Gamma:float,t_list:list,c:float,pk_list:list,P:float) -> list:

    dyn_list = []
    for t_index in progressbar(np.arange(len(t_list))):
        time = t_list[t_index]
        ck_list = compute_bath_total_analytical_single_time(initial_state_dic,ws,Gamma,time,c,pk_list,P)
        
        dyn_list.append(ck_list)
    return dyn_list