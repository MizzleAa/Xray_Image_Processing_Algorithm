from library.utils.header import *
from library.utils.decorator import *

def crop_rectangle(data, options={}):
    '''
    options = {
        "x" : {x},
        "y" : {y},
        "width" : {width},
        "height" : {height},
    }
    '''
    x = int(options["x"])
    y = int(options["y"])
    width = int(options["width"])
    height = int(options["height"])
    
    result = data[y:y+height,x:x+width]
    return result

def crop_list_rectangle(data, options={}):
    '''
    options = {
        "x" : {x},
        "y" : {y},
        "width" : {width},
        "height" : {height},
    }
    '''
    x = int(options["x"])
    y = int(options["y"])

    width = int(options["width"])
    height = int(options["height"])
    
    cut_list = cutting(x,y,width,height)
    data_list = []
    for cut in cut_list:
        tmp_data = copy.copy(data)
        max_size = np.iinfo(tmp_data.dtype).max
        tmp_data[cut[1]:cut[3],cut[0]:cut[2]] = max_size
        data_list.append(tmp_data)
    
    return data_list, cut_list

def cutting(x,y,width,height):
    '''
    x : 
    y :
    width :
    height : 
    '''
    rate = 5
    result = [
        [x,y,x+width,y+(height//rate)],
        [x,y+((rate-1)*height//rate ),x+width,y+height],
        [x,y,x+(width//rate),y+height],
        [x+((rate-1)*width//rate ),y,x+width,y+height],
    ]
    return result

