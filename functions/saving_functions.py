import pickle
import os

def save_dict(var_dict,data_file,overwrite=True,vb=False):
    if vb: print("Saving ",list(var_dict.keys()), "in", data_file,".")
    else: print("Saving Data. ")
    if os.path.isfile(data_file):
        old_dict = load_dict(data_file,vb=False)
        new_dict = old_dict.copy()
        keys = list(var_dict)
        for key in keys:
            if key in new_dict:
                if vb: print("Variable with same name already saved.")
                if overwrite:
                    if vb: print("Overwriting.")
                    new_dict[key] = var_dict[key]
                else:
                    if vb: print("Skipping.")
            else:
                new_dict[key] = var_dict[key]

        with open(data_file,'wb') as pickle_out:
            pickle.dump(new_dict,pickle_out)
            pickle_out.close()
    else:
        pickle_out = open(data_file,'wb')
        pickle.dump(var_dict,pickle_out)
        pickle_out.close()
    print('Finished Saving Data.')


def load_dict(data_file,vb=False):
    print('Loading Data.')
    with open(data_file,'rb') as pickle_in:
        pickled_dict = pickle.load(pickle_in)
        pickle_in.close()
    if vb: print("Finished Loading Data:", list(pickled_dict.keys()))
    else: print("Finished Loading Data.")
    return pickled_dict

