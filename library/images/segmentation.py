from library.utils.header import *
from library.utils.decorator import *

def segmentation(data, options={}):
    '''
    options = {
        "min_area":{min_area},
        "cliping_dtype":{cliping_dtype},
        "threshold":{threshold},
        "max_pixel":{max_pixel}
    }
    '''
    min_area = options["min_area"]
    cliping_dtype = options["cliping_dtype"]
    threshold = options["threshold"]
    max_pixel = options["max_pixel"]

    width, height = data.shape

    tmp_data = np.where(
        data > threshold,
        0,
        max_pixel
    )
    tmp_data = np.asarray(tmp_data, dtype="uint8")
    '''
        RETR_CCOMP: int
        RETR_EXTERNAL: int
        RETR_FLOODFILL: int
        RETR_LIST: int
        RETR_TREE: int

        CHAIN_APPROX_NONE: int
        CHAIN_APPROX_SIMPLE: int
        CHAIN_APPROX_TC89_KCOS: int
        CHAIN_APPROX_TC89_L1: int
    '''
    contours, hierarchy = cv2.findContours(
        tmp_data, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    ctrs = []

    for contour in contours:
        if min_area < cv2.contourArea(contour) < (width-10)*(height-10):
            ctrs.append(contour)

    masks = []

    for ctr in ctrs:
        mask = np.zeros(tmp_data.shape, dtype="uint8")
        mask = cv2.fillConvexPoly(mask, ctr, 255)
        masks.append(mask)

    slices = []

    for ctr, mask in zip(ctrs, masks):
        x_min = np.min(ctr[:, 0, 0])
        x_max = np.max(ctr[:, 0, 0])

        y_min = np.min(ctr[:, 0, 1])
        y_max = np.max(ctr[:, 0, 1])

        mask = np.where(
            mask == 255,
            1,
            0
        )

        mask = np.asarray(mask, dtype=cliping_dtype)

        clip = np.multiply(data, mask)
        slc = np.asarray(clip[y_min:y_max, x_min:x_max], cliping_dtype)

        slices.append(slc)

    return ctrs, masks, slices