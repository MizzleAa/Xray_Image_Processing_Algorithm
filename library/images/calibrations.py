from library.utils.header import *
from library.utils.decorator import *

def calibrations(data, options={}):
    '''
    data
    options = {
        "padding":{padding},
        "mean_height":{mean_height}
    }
    '''
    padding = options["padding"]
    mean_height = options["mean_height"]

    max_depth = np.iinfo(data.dtype).max
    '''
    # 외각라인 제거
    # 크기를 바꿔줌 padding
    # 1000x1000 = (1000-padding)x(1000-padding)
    '''
    data = data[padding:-padding, padding:-padding]

    '''
    # 조건에 따른 결과값 출력
    # 역치 값의 그래프 뒤집는 효과
    # if 문
    '''
    # 평횔비 적용
    data = np.where(
        data > max_depth/2,
        data - max_depth/2,
        data + max_depth/2
    )
    '''
    # 평균값 계산
    '''
    val = np.mean(data[0:mean_height], axis=0)
    # 평균, 표준편차를 통한 % 값 계산
    '''
    # 평균 값들의 평균
    '''
    mean = np.mean(val)
    percent = (val-mean)/mean
    '''
    # calibration
    '''
    data = val - data
    '''
    # 감쇠비율 조정
    '''
    data = max_depth - (data - data * percent)
    '''
    if data > max_depth:
        data = max_depth
    else:
        data = data
    '''
    data[data > max_depth] = max_depth
    data[data < 0] = 0

    # norm = np.linalg.norm(data)
    # data = data/norm * 1000 * 65535
    '''
    평횔비 : 히스토그램
    '''
    data = (data - np.min(data))/(np.max(data)-np.min(data)) * max_depth
    return data
