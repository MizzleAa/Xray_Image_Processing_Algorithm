from library.utils.header import *
from library.utils.decorator import *

def sharp(self, data, level):
    kernel = Kernel.sharp(level)
    
    result = cv2.filter2D(data, ddepth=-1, kernel=kernel)
    result = result.astype(data.dtype)

    return result

class Kernel:
    @staticmethod
    def sharp(level):
        x = np.full((3,3),-1)
        x[1,1] = 9+level-1
        return x