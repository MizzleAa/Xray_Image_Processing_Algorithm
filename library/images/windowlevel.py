from library.utils.header import *
from library.utils.decorator import *
from library.utils.io import *

@dec_func_start_end
def windowlevel_make():
    
    window_level = WindowLevel()
    origin_path = "F://ct/_analysis_/origin"
    save_path = "F://ct/_analysis_/windowlevel"
    
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
            
            options = {
                "load_path":f"{path}",
                "file_name":f"{name}",
                "save_file_name":f"{save_path}/{object}/image/{name}",
                "algorithm":"linear_1_6000"
            }
            make_dir(f"{save_path}/{object}/image")
            # print(options)
            window_level.run(options)

class WindowLevel:
    def __init__(self) -> None:
        pass
    
    def run(self, options):
        load_path = options["load_path"]
        file_name = options["file_name"]
        save_file_name = options["save_file_name"]
        algorithm = options["algorithm"]
            
        image_options = {
            "file_name": f"{load_path}/{file_name}",
            "dtype": np.uint16
        }
        data = load_image(image_options)
        if algorithm == "linear_1_4000":
            result = self._linear(data,1.0,4000)
        if algorithm == "linear_1_6000":
            result = self._linear(data,1.0,6000)
        if algorithm == "linear_1_8000":
            result = self._linear(data,1.0,8000)
        if algorithm == "linear_1_16000":
            result = self._linear(data,1.0,16000)
        if algorithm == "linear_1.5":
            result = self._linear(data,1.5,0)
        if algorithm == "linear_2.0":
            result = self._linear(data,2.0,0)
        if algorithm == "linear_2.5":
            result = self._linear(data,2.5,0)
        if algorithm == "sqrt":
            result = self._sqrt(data,1.0)
        if algorithm == "power":
            result = self._power(data,1.0)
        if algorithm == "sqrt_1.5":
            result = self._sqrt(data,1.5)
        if algorithm == "power_1.5":
            result = self._power(data,1.5)
            
        save_options = {
            "file_name": f"{save_file_name}",
            "dtype": np.uint16,
            "start_pixel" : 0,
            "end_pixel" : 65535
        }
        save_image(result, save_options)
        
    def _linear(self, data, x=1.5, y=0):
        float_data = data / 65535
        result = x*float_data
        result = result * 65535 +y
        return result
    
    def _sqrt(self, data, x=1.0):
        float_data = data / 65535
        result = x*np.sqrt(float_data)
        result = result * 65535
        return result
    
    def _power(self, data, x=1.0):
        float_data = data / 65535
        result = x*np.power(float_data,2)
        result = result * 65535
        return result
    
def windowlevel(options={}):
    load_path = options["load_path"]
    file_name = options["file_name"]
    save_file_name = options["save_file_name"]
    
    image_options = {
        "file_name": f"{load_path}/{file_name}",
        "dtype": np.uint16
    }
    data = load_image(image_options)
    float_data = data / 65535
    linear_data = 1.5*float_data
    sqrt_data = np.sqrt(float_data)
    squared_data = np.power(float_data,2)
    
    linear_data = linear_data * 65535
    sqrt_data = sqrt_data * 65535
    squared_data = squared_data * 65535
    
    save_options = {
        "file_name": f"{load_path}/{save_file_name}",
        "dtype": np.uint16,
        "start_pixel" : 0,
        "end_pixel" : 65535
    }
    save_options["file_name"] = f"{load_path}/linear_{save_file_name}"
    save_image(linear_data, save_options)
    
    save_options["file_name"] = f"{load_path}/sqrt_{save_file_name}"
    save_image(sqrt_data, save_options)
    
    save_options["file_name"] = f"{load_path}/squared_{save_file_name}"
    save_image(squared_data, save_options)
    
    # plt.imshow(squared_data)
    # plt.show()
    # plt.imshow(sqrt_data)
    # plt.show()
    # plt.imshow(float_data)
    # plt.show()
    #save_image(result, save_options)
    