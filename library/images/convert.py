from library.utils.header import *
from library.utils.decorator import *
from library.utils.io import *
from library.utils.progress import *

@dec_func_start_end
def xray_to_color_make():
    options = {
        "load_path":"E://daq/_analysis_/xray_to_color",
        "high_file_name":"high.png",
        "low_file_name":"low.png",
        "save_file_name":"save.png"
    }
    xray_to_color(options)

@dec_func_start_end
def image_8bit_to_16bit_make():
    # origin_path = "F://custom/_source_/origin/xray_origin"
    # save_path = "F://custom/_source_/origin/xray_origin_gray"
    
    origin_path = "./sample/xray/example_10/convert"
    save_path = "./sample/xray/example_10/result"

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
        
        for path, name in zip(paths, names):
            object = path.split("/")[-2]
            
            image_options = {
                "file_name": f"{path}/{name}",
                "dtype": np.uint8
            }
            data = load_image(image_options)
            color_to_gray = image_color_to_gray(data)
            
            file_name = name.split(".")[0]
            make_dir(f"{save_path}/{object}/image")
    
            save_options = {
                "file_name": f"{save_path}/{object}/image/{file_name}.png",
                "dtype": np.uint8,
                "start_pixel" : 0,
                "end_pixel" : 255
            }
            save_image(color_to_gray, save_options)
            
def xray_to_png(options={}):
    '''
    options = {
        "load_path":{load_path},
        "file_name":{file_name},
        "height":{height},
        "width":{width},
    }
    '''
    load_path = options["load_path"]
    file_name = options["file_name"]
    
    height = options["height"]
    width = options["width"]

    file = open(os.path.join(load_path, file_name), 'rb')
    file_to_array = np.fromfile(file, dtype=np.uint8, count=height * width * 2)
    raw_image = np.reshape(file_to_array, (height, width, 2))

    image_high = raw_image[:, :, 0] * 256
    image_low = raw_image[:, :, 1]

    result = image_high + image_low

    return result

def ct_to_png(options={}):
    '''
    options = {
        "load_path":{load_path},
        "file_name":{file_name},
        "height":{height},
        "width":{width},
    }
    '''

    load_path = options["load_path"]
    file_name = options["file_name"]
    
    height = options["height"]
    width = options["width"]
    
    file = open(os.path.join(load_path, file_name), 'rb')
    file_to_array = np.fromfile(file, dtype=np.uint16, count=height * width)
    raw_image = np.reshape(file_to_array, (height, width))
    result = raw_image

    return result


def image_color_to_gray(data):
    result = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return result

def image_color_to_16bit(data):
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
    gray = 0.2989 * b + 0.5870 * g + 0.1140 * r
    result = gray/255 * 65535
    # result = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return result

def image_8bit_to_16bit(data):
    data = np.mean(data, axis=2)
    data = data * 255
    return data

def xray_to_color(options={}):
    #TODO 개발 진행중 
    
    load_path = options["load_path"]
    high_file_name = options["high_file_name"]
    low_file_name = options["low_file_name"]
    save_file_name = options["save_file_name"]
    
    image_options = {
        "file_name": f"{load_path}/{high_file_name}",
        "dtype": np.uint16
    }
    high_data = load_image(image_options)
    
    image_options = {
        "file_name": f"{load_path}/{low_file_name}",
        "dtype": np.uint16
    }
    low_data = load_image(image_options)
    
    hight, width = np.shape(high_data)
    
    result = np.full((hight, width, 3), 0)
    
    for h in range(hight):
        for w in range(width):
            value = sample_graph(high_data[h][w], low_data[h][w])   
            result[h][w] = value
            
    save_options = {
        "file_name": f"{load_path}/{save_file_name}",
        "dtype": np.uint16,
        "start_pixel" : 0,
        "end_pixel" : 65535
    }
    plt.imshow(result)
    plt.show()
    #save_image(result, save_options)

def xray_container_32bit_to_16bit(data, options={}):
    fix_width = options["fix_width"]
    
    max_depth = np.iinfo(np.int32).max
    
    data[data==0] = np.max(data)
    data = (data - np.min(data))/(np.max(data)-np.min(data)) * np.iinfo(np.uint16).max
    data[data==np.max(data)] = 0
    data = data.astype(np.uint16)
    
    return data
    
def sample_graph(high, low):
    # TODO : 수식 알고리즘 수행
    y_axis_list = [-2**16,2**0,2**8,2**16]
    color_list = [ [0,0,0],[255,0,0],[0,255,0],[255,255,255]]
    
    result = [255,255,255] 
    
    for y_axis, color in zip(y_axis_list,color_list):
        if - low + y_axis > high:
            # print(y_axis,low , high)
            result = color
            break
    # print(result)
    return result