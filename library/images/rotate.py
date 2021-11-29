from library.utils.header import *
from library.utils.decorator import *

def rotate(data, options={}):
    '''
    options = {
        "angle": {angle}
    }
    '''
    angle = options["angle"]

    (height, width) = data.shape[:2]
    (center_x, center_y) = (width // 2, height // 2)
    # 회전
    rotate_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    # 극좌표
    cos = np.abs(rotate_matrix[0, 0])
    sin = np.abs(rotate_matrix[0, 1])
    # 이미지 크기 재설정
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    # 중점설정
    rotate_matrix[0, 2] += (new_width // 2) - center_x
    rotate_matrix[1, 2] += (new_height // 2) - center_y
    # 그리기
    # print(np.shape(data))
    result = cv2.warpAffine(data, rotate_matrix, (new_width, new_height))

    return result, rotate_matrix