from library.utils.header import *
from library.utils.decorator import *

def slice(data, options={}):
    '''
    options = {
        "padding":{padding},
        "limit_size":{limit_size},
    }
    '''
    padding = options["padding"]
    limit_size = options["limit_size"]

    max_depth = np.iinfo(data.dtype).max
    
    data = data[padding:-padding, padding:-padding]
    line_data = data.T

    val = np.mean(data, axis=0)
    
    #print(val)

    tmp = []
    result = []
    for key, value in enumerate(val):
        if value < max_depth:
            tmp.append(line_data[key])
        else:
            if len(tmp) > limit_size:
                result.append(np.array(tmp))
                tmp = []

    result.append(np.array(tmp))
    #print(result)

    return result