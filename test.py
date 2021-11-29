from library.utils.header import *

def rotate(data, angle):
    '''
    options = {
        "angle": {angle}
    }
    '''

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

def mask_filter_background(data, segmentation):
    
    height, width = np.shape(data)
    mask = np.full((height,width),0)
    
    segmentation = segmentation[0]
    len_segmentation = (len(segmentation))//2
    segmentation = np.array(segmentation, dtype=np.int32).reshape((len_segmentation,2))
    cv2.fillPoly(mask, [segmentation], 1, cv2.LINE_AA)
    
    mask_16 = np.array(mask, dtype=np.uint16)
    
    result = np.full((height,width),65535, dtype=np.uint16)
    result = np.where(
        mask_16 == 1,
        data,
        result
    )
    result[result==0] = 65535
    return result


def opencv_rotate_test():
    load_file_name = "./sample/16bit.png"
    
    image = cv2.imread(load_file_name,-1)
    rotate_image, rotate_matrix = rotate(image, 30)
    
    cv2.imshow('origin', image)
    cv2.imshow('rotate_image', rotate_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def opencv_remap_test():
    load_file_name = "./sample/16bit.png"
    
    image = cv2.imread(load_file_name,-1)
    rows, cols = image.shape[:2]

    exp = 0.8
    scale = 1.2
    
    map_y, map_x = np.indices((rows, cols),dtype=np.float32)

    map_x = 2*map_x/(cols-1)-1
    map_y = 2*map_y/(rows-1)-1
    
    r, theta = cv2.cartToPolar(map_x, map_y)

    r[r< scale] = r[r<scale] **exp  
    map_x, map_y = cv2.polarToCart(r, theta)
    
    map_x = ((map_x + 1)*cols-1)/2
    map_y = ((map_y + 1)*rows-1)/2

    distorted = cv2.remap(image,map_x,map_y,cv2.INTER_LINEAR)

    cv2.imshow('origin', image)
    cv2.imshow('remap', distorted)
    cv2.waitKey()
    cv2.destroyAllWindows()

def opencv_affine_test():
    load_file_name = "./sample/16bit.png"
    
    image = cv2.imread(load_file_name,-1)

    aff = np.array([[1, 0.09, 0],
                    [0.01, 0.5, 0]], dtype=np.float32)

    rows, cols = image.shape[:2]

    distorted = cv2.warpAffine(image, aff, (cols , rows))
    
    cv2.imshow('origin', image)
    cv2.imshow('affine', distorted)
    cv2.waitKey()
    cv2.destroyAllWindows()

# opencv_remap_test()
# opencv_affine_test()
# opencv_rotate_test()

