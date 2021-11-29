from library.utils.header import *
from library.utils.decorator import *

def mask(data, options={}):
    '''
    options = {
        "threshold":{threshold},
        "max_pixel":{max_pixel},
    }
    '''
    threshold = options["threshold"]
    max_pixel = options["max_pixel"]

    result = np.where(
        data > threshold,
        0,
        max_pixel
    )
    return result