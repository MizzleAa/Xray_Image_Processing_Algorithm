from library.utils.header import *
from library.utils.decorator import *

from library.utils.io import *
from library.utils.progress import *

@dec_func_start_end
def blur_run():
    
    origin_path = "F://ct/_analysis_/windowlevel"
    save_path = "F://ct/_analysis_/blur"
    
    dir_list_options = {
        "dir_path": origin_path
    }
    
    fulls, _, _ = load_dir_list(dir_list_options)
    
    make_dir(f"{save_path}",{"is_remove":True})
    
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
            blur_data = blur(data)
            
            make_dir(f"{save_path}/{object}/image")
            
            save_image_options = {
                "file_name": f"{save_path}/{object}/image/{name}",
                "dtype": np.uint16,
                "start_pixel": 0,
                "end_pixel": 65535
            }
            
            # print(save_image_options["file_name"])
            save_image(blur_data,save_image_options)

def blur(data, options={}):
    kernel = np.array(
        [
            [0.0,1.0,0.0], 
            [1.0,2.0,1.0], 
            [0.0,1.0,0.0]
        ]
    )
    kernel = kernel / np.sum(kernel)
    # kernel = kernel / np.sum(kernel)
    array_list = []
    
    for y in range(3):
        copy_array = np.copy(data)
        copy_array = np.roll(copy_array, y - 1, axis=0)
        for x in range(3):
            copy_array_x = np.copy(copy_array)
            copy_array_x = np.roll(copy_array_x, x - 1, axis=1)*kernel[y,x]
            array_list.append(copy_array_x)

    array_list = np.array(array_list)
    array_list_sum = np.sum(array_list, axis=0)
    return array_list_sum