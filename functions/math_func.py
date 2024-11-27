import numpy as np

def coth(beta,x):
    if x == 0:
        raise Exception('argument should not be zero')
    if beta == 'inf':
        return sg(np.real(x))
    return (np.exp(beta*x) + np.exp(-beta*x)) / (np.exp(beta*x) - np.exp(-beta*x))
def sg(t):
    if t > 0:
        return 1
    return -1
def theta(t):
    if t < 0:
        return 0
    return 1
def gaussian(x,mu,sigma):
    return np.exp(- (x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2*np.pi) * sigma)
def fourier(fk_list:list,pk_list:list,x_list:list) -> list:
    Delta_p = pk_list[1] - pk_list[0]
    fx_list = []
    for x in x_list:
        res = 0
        for fk,pk in zip(fk_list,pk_list):
            res += Delta_p * fk * np.exp(1j * pk * x)
        fx_list.append(res / np.sqrt(2 * np.pi))
    return fx_list
