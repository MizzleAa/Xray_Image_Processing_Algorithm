from library.utils.header import *
from library.utils.decorator import *

def filp(data, options={}):
    '''
    options = {
        "filp":{filp}
    }
    '''
    filp = options["filp"]

    # 반전
    # 1 : 좌우 
    # 0 : 상하 
    # -1 : 좌우 & 상하
    result = cv2.flip(data, filp)
    
    return result
