from library.utils.header import *
from library.utils.decorator import *
from library.utils.io import *
from library.utils.progress import *


@dec_func_start_end
def image_crop_make():
    # origin_path = "E:/police/구두/8bit"
    # save_path = "E:/police/구두/16bit"
    
    origin_path = "./sample/xray/example_15/16bit"
    save_path = "./sample/xray/example_15/crop"

    dir_list_options = {
        "dir_path": origin_path
    }
    
    fulls, _, _ = load_dir_list(dir_list_options)
    
    make_dir(f"{save_path}",{"is_remove":False})
    
    progress = Progress(max_num=len(fulls),work_name=__name__)
    
    for full in fulls:
        #progress.set_work_name(f" = {full}\n")
        progress.update()

        options = {
            "ends_with": (".jpg", ".png"),
            "file_path": f"{full}/image"
        }
        
        _, paths, names = load_file_list(options)
        
        for index, (path, name) in enumerate(zip(paths, names)):
            object = path.split("/")[-2]
            
            image_options = {
                "file_name": f"{path}/{name}",
                "dtype": np.uint16
            }
            data = load_image(image_options)
            #color_to_gray = image_color_to_gray(data)
            #data = image_8bit_to_16bit(data)
            height, width = data.shape[:2]
            
            crop_padding = 300
            data = crop_rectangle(data, options={
                "x":crop_padding,
                "y":0,
                "width":(width-crop_padding*2),
                "height":height
            })
            # print("\n",np.shape(data), data.dtype , np.min(data), np.max(data))

            #file_name = name.split(".")[0]
            make_dir(f"{save_path}/{object}/image")
    
            save_options = {
                "file_name": f"{save_path}/{object}/image/{index}.png",
                "dtype": np.uint16,
                "start_pixel" : 0,
                "end_pixel" : 65535
            }
            # print(save_options)
            # save_image(data, save_options)
            cv_save_image(data, save_options)



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

