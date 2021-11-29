from library.utils.header import *
from library.utils.decorator import *

def noise(options={}):
    '''
    options = {
        "min_value":{min_value},
        "max_value":{max_value},
        "width":{width},
        "height":{height},
    }
    '''
    min_value = options["min_value"]
    max_value = options["max_value"]
    width = options["width"]
    height = options["height"]

    result = np.random.randint(min_value,max_value,(height,width))

    return result
