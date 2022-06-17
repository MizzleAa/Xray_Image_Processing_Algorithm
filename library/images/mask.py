from library.utils.header import *
from library.utils.decorator import *


def mask_polygon(data, segmentation, options={}):
    seg_list = []
    
    segmentation = segmentation[0]
    for i in range(0,len(segmentation),2):
        seg_list.append([segmentation[i],segmentation[i+1]])
        
    result = copy.copy(data)
    if len(result.shape) == 3:
        result = result[:,:,0]
    height, width = result.shape[:2]
    
    mask = np.full( (height, width), 255, dtype = np.int32)
    segmentation = np.array(seg_list, dtype = np.int32)

    cv2.fillPoly(mask, [segmentation], 1)
    result = result * mask
    #result[result==0]=np.iinfo(result.dtype).max
    return result

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