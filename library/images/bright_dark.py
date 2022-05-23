from library.utils.header import *
from library.utils.decorator import *

from library.utils.io import *
from library.utils.progress import *

def bright_dark(data, options={}):
    option_level = options["level"]
    data_max = np.iinfo(data.dtype).max
    data_min = np.iinfo(data.dtype).min
    
    float_data = data / data_max
    result = float_data + option_level
    result = result * data_max
    result[result > data_max] = data_max
    result[result < data_min] = data_min
    
    result= result.astype(data.dtype)
    return result