from library.utils.header import *
from library.utils.decorator import *

def reverse(data):
    max_value = np.iinfo(data.dtype).max
    result = max_value - data
    return result