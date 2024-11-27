import numpy as np

def adjust(t_out_list,t_list):
    # replaces the elements of the first list with the corresponding closest from the second list 
    t_out_list_adj = []
    for t_out in t_out_list:
        n_adj = abs(np.array(t_list)-t_out).argmin()
        t_out_adj = t_list[n_adj]
        t_out_list_adj.append(t_out_adj)
    return t_out_list_adj

def partial_list(t_list,t_out):
    if t_out == 0:
        res = [0,t_out]
        return res
    res = []
    for t in t_list:
        if t <= t_out:
            res.append(t)
    return res