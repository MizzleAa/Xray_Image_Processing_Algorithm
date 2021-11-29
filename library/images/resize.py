from library.utils.header import *
from library.utils.decorator import *

from library.utils.io import *
from library.utils.progress import *

from library.images.backgroundfix import *

@dec_func_start_end
def resize_run():
    # origin_path = "E://aixac/_background_/null/origin"
    # save_path = "E://aixac/_background_/null/background_fix"

    # origin_path = "E://daq/_verify_/xray_hard_gasgun_a/image"
    # save_path = "E://daq/_verify_/background_fix"
    
    origin_path = "F://ct/_analysis_/origin"
    save_path = "F://ct/_analysis_/resize"
    
    dir_list_options = {
        "dir_path": origin_path
    }
    
    fulls, _, _ = load_dir_list(dir_list_options)
    
    make_dir(f"{save_path}",{"is_remove":True})
    
    resize_options = {
        "max_width":700,
        "max_height":570
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
            
            resize_options = {
                "width":700,
                "height":570
            }
            data = load_image(load_image_options)
            background_fix_data = background_fix_center(data, resize_options)
            
            make_dir(f"{save_path}/{object}/image")
            
            save_image_options = {
                "file_name": f"{save_path}/{object}/image/{name}",
                "dtype": np.uint16,
                "start_pixel": 0,
                "end_pixel": 65535
            }
            
            # print(save_image_options["file_name"])
            save_image(background_fix_data,save_image_options)

def cv_resize(data, options={}):
    '''
    options = {
        "width":{width},
        "height":{height},
        "fx":{fx},
        "fy":{fy},
    }
    '''
    width = options["width"]
    height = options["height"]
    fx = options["fx"]
    fy = options["fy"]
    
    result = cv2.resize(data, dsize=(width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return result

def resize(data, options):
    '''
    options = {
        "width":{width},
        "height":{height},
    }
    '''
    width = options["width"]
    height = options["height"]
    cols, rows = np.shape(data)

    return [[ data[int(cols * c / height)][int(rows * r / width)]  
                for r in range(width)] for c in range(height)]
