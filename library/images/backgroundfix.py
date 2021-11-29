from library.utils.header import *
from library.utils.decorator import *

from library.utils.io import *
from library.utils.progress import *

@dec_func_start_end
def background_null_make():
    save_path = "E://daq/_background_/null/none"
    
    make_dir(save_path,{"is_remove":True})

    background_fix_options = {
        "max_width":1200,
        "max_height":1156
    }
    background_fix_data = background_fix_null(background_fix_options)

    save_image_options = {
        "file_name": f"{save_path}/0.png",
        
        "dtype": np.uint16,
        "start_pixel": 0,
        "end_pixel": 65535
    }
    save_image(background_fix_data,save_image_options)
    
@dec_func_start_end
def background_fix_run():
    # origin_path = "E://aixac/_background_/null/origin"
    # save_path = "E://aixac/_background_/null/background_fix"

    # origin_path = "E://daq/_verify_/xray_hard_gasgun_a/image"
    # save_path = "E://daq/_verify_/background_fix"
    
    # origin_path = "F://ct/_analysis_/origin"
    # save_path = "F://ct/_analysis_/origin_background_fix"
    
    origin_path = "F://custom/_seperation_/drug_reside"
    save_path = "F://custom/_seperation_/drug_reside_background_fix"
    
    dir_list_options = {
        "dir_path": origin_path
    }
    
    fulls, _, _ = load_dir_list(dir_list_options)
    
    make_dir(f"{save_path}",{"is_remove":True})
    
    background_fix_options = {
        "max_width":1200,
        "max_height":1156
    }

    for full in fulls:
        options = {
            "ends_with": ".png",
            "file_path": f"{full}/image"
        }
        _, paths, names = load_file_list(options)
        for path, name in zip(paths, names):
            
            object = path.split("/")[-2]
            load_image_options = {
                "file_name": f"{full}/image/{name}",
                "dtype": np.uint16
            }
            data = load_image(load_image_options)
            background_fix_data = background_fix_center(data, background_fix_options)
            
            make_dir(f"{save_path}/{object}/image")
            
            save_image_options = {
                "file_name": f"{save_path}/{object}/image/{name}",
                "dtype": np.uint16,
                "start_pixel": 0,
                "end_pixel": 65535
            }
            
            # print(save_image_options["file_name"])
            save_image(background_fix_data,save_image_options)


def background_fix_null(options={}):
    max_width = options["max_width"]
    max_height = options["max_height"]
    
    result = np.full((max_height, max_width), 65535)

    return result


def background_fix_center(data, options={}):
    '''
    data,
    options = {
        "max_width": {max_width},
        "max_height": {max_height}
    }
    '''

    max_width = options["max_width"]
    max_height = options["max_height"]
    
    height, width = data.shape[:2]
    
    if max_width < width:
        max_width = width 
        # print(width, height, max_width, max_height)

    if max_height < height:
        max_height = height 
        # print(width, height, max_width, max_height)
    result = np.full((max_height, max_width), 65535)
        
    tmp_width = int(width // 2)
    tmp_height = int(height // 2)

    w_min = int(max_width // 2) - tmp_width
    h_min = int(max_height // 2) - tmp_height
    
    result[h_min:h_min + height, w_min:w_min + width] = data

    return result

def background_fix_position(data, options):
    '''
    data,
    options = {
        "max_width": {max_width},
        "max_height": {max_height},
        "point_x":{point_x},
        "point_y":{point_y},
    }
    '''
    
    max_width = options["max_width"]
    max_height = options["max_height"]

    point_x = options["point_x"]
    point_y = options["point_y"]

    height, width = data.shape[:2]
    result = np.full((max_height, max_width), 65535)

    result[point_y:point_y + height, point_x:point_x + width] = data

    return result
