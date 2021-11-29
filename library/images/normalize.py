from library.utils.header import *
from library.utils.decorator import *

def normalize_adjust(data, options={}):
    '''
    options = {
        "alpha" : {alpha},
        "beta" : {beta},
        "gamma" : {gamma},
    }
    '''
    alpha = float(options["alpha"])
    beta = float(options["beta"])
    gamma = float(options["gamma"])

    result = alpha * ( data - beta ) + gamma
    #self.console(result)
    return result

def normalize_equalization(data):
    max_depth = np.iinfo(data.dtype).max
    result = (data - np.min(data))/(np.max(data)-np.min(data))* max_depth
    return result
