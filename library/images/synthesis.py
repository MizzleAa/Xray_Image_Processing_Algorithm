from library.utils.header import *
from library.utils.decorator import *

def synthesis(data_a, data_b, options={}):
    '''
    options = {
        "alpha":{alpha},
        "beta":{beta},
        ""
    }
    '''
    alpha = options["alpha"]
    beta = options["beta"]
    
    max_pixel = 65535
    
    alpha_data = max_pixel - alpha*data_a
    beta_data = max_pixel - beta*data_b

    result = alpha_data + beta_data
    return max_pixel-result