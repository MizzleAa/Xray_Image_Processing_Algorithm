from library.utils.header import *
from library.utils.decorator import *

def null_check(data, options={}):
    padding = options["padding"]
    mean_height = options["mean_height"]
    limit_std_size = options["limit_std_size"]
    limit_min_size = options["limit_min_size"]
    is_line = options["is_line"]
    max_depth = np.iinfo(data.dtype).max
    
    if not is_line:
        data = data.T
    data = data[padding:-padding, padding:-padding]

    val = np.mean(data[0:mean_height], axis=0)
    mean = np.mean(val)
    percent = (val-mean)/mean
    data = val - data
    data = max_depth - (data - data * percent)
    data[data > max_depth] = max_depth
    data[data < 0] = 0

    is_null = False
    data_std = np.std(data)
    data_min = np.min(data)
    if data_std < limit_std_size and limit_min_size < data_min:
        is_null = True

    return is_null
