from library.utils.header import *
from library.utils.decorator import *

from library.utils.io import *
from library.utils.progress import *

def contrast(data, options={}):
    level = options["level"]
    max_data = np.iinfo(data.dtype).max
    min_data = np.iinfo(data.dtype).min
    middle = (max_data + min_data) / 2
    result = np.clip( (data)*(1 + level/10), min_data, max_data)
    result = result.astype(data.dtype)
    return result