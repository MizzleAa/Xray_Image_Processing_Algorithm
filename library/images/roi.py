from library.utils.header import *
from library.utils.decorator import *

def roi(data, options={}):
    option_roi = options['roi']
    data_bins = np.iinfo(data.dtype).max
    min_data =np.iinfo(data.dtype).min
    
    if option_roi:
        x, y, width, height = option_roi
        roi = data[y: y + height, x: x + width]
    else:
        roi = data
    
    #hist 노말라이즈 뽑고
    hist, bins = np.histogram(roi, data_bins, [0,data_bins])
    
    #누적계수 뽑고
    cumsum = hist.cumsum()
    #누적 계수에 대한 평균값을 구함
    cumsum = data_bins * cumsum / cumsum[-1]
    
    #히스토그램 평활화 수행
    equalized = np.interp(data.flatten(), bins[:-1], cumsum)
    result = equalized.reshape(data.shape)
    result = result.astype(data.dtype)
    return result