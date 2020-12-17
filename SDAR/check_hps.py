import pickle
import os

exper_dir = 'experiments/param_search'
load_name = '3.3__num_spline'
# load_name1 = '1.1_lhd5__num_spline'
load_dir = os.path.join(exper_dir, load_name)
with open(load_dir, 'rb') as f:
    sp = pickle.load(f)
    res = pickle.load(f)

# load_dir1 = os.path.join(exper_dir, load_name1)
# with open(load_dir1, 'rb') as f:
#     sp1 = pickle.load(f)
#     res = pickle.load(f)
