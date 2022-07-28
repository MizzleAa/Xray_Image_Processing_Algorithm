from library.utils.header import *
from library.utils.io import *

def histogram(data):
    '''
    #result = cv2.equalizeHist(data)
    #return result
    '''
    height, width = data.shape[0:2]
    min_value = np.min(data)
    max_value = np.max(data)
    print(min_value, max_value)
    return data

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
    
def opencv_histogram_test():
    load_file_name = "./sample/xray/example_7/image/20220405161005_lupin_v4_00070_side.jpg"
    
    #image = cv2.imread(load_file_name,cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(load_file_name,cv2.IMREAD_UNCHANGED)
    histogram_image = histogram(image)
    
    cv2.imshow('origin', image)
    cv2.imshow('histogram', histogram_image)
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

def spectrum_map_v1():
    
    width = 255
    height = 255
    
    data = np.full(shape=(height,width,3),fill_value=0.0)
    
    split_hip = height*math.sqrt(2)
    
    for i in range(height):
        for j in range(width):
            cal = 1.0-math.sqrt( i*i+j*j ) / (split_hip)
            
            data[height-i-1][width-j-1] = cal
            # data[height-i-1][width-j-1][0] = cal_blue
            # data[height-i-1][width-j-1][1] = cal_green
            # data[height-i-1][width-j-1][2] = cal_red
            
    cv2.imshow('spectrum_map', data)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return data

def spectrum_run():
    s_map = spectrum_map_v1()
    
    low_load_file_name = "./sample/xray/example_3/low.png"
    high_load_file_name = "./sample/xray/example_3/high.png"
    
    low = cv2.imread(low_load_file_name,-1)
    high = cv2.imread(high_load_file_name,-1)

    height, width = np.shape(low)
    
    data = np.zeros(shape=(height,width,3), dtype=np.float32)
    
    for h in range(height):
        for w in range(width):
            l_value = low[h][w]
            h_value = high[h][w]

            # l_value = l_value // 256 - 1
            # h_value = h_value // 256 - 1
            #data[h][w] = s_map[l_value][h_value]

            # l_value = l_value // 256 - 1
            # h_value = h_value // 256 - 1
            
            value = h_value/l_value # 기울기
            
            data[h][w] = value * 255
            #print(value, h_value,l_value)
    
    print(np.max(data), np.min(data))
    #print(data)
    
    cv2.imshow('spectrum_run', data)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("./sample/xray/example_3/result.jpg", data)
    
def cv_imread_color_test():
    high_load_file_name = "./sample/xray/example_3/high.png"
    high = cv2.imread(high_load_file_name,cv2.IMREAD_LOAD_GDAL)
    # print(high)
    cv2.imshow('spectrum_run', high)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cv_imread_16bit_3channel():
    # high_load_file_name = "./sample/xray/example_3/high.png"
    high_load_file_name = "./sample/ai/example_1/taser_gun/image/00108.png"
    
    high = cv2.imread(high_load_file_name,cv2.IMREAD_UNCHANGED)
    
    data1 = np.array(high//256, dtype=np.uint8)
    data2 = np.array(high%256, dtype=np.uint8)
    data3 = np.array(high//65535*255, dtype=np.uint8)
    
    data = cv2.merge((data1, data2, data3))
    
    cv2.imshow('data1', data1)
    cv2.imshow('data2', data2)
    cv2.imshow('data3', data3)
    cv2.imshow('data', data)
    
    cv2.waitKey()
    cv2.destroyAllWindows()

import argparse
def argparse_example():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--images", required=True, type=str,
		help="path to input directory of images")
	ap.add_argument("-o", "--output", required=True, type=str,
		help="path to output directory to store intermediate files")
	ap.add_argument("-a", "--hashes", required=True, type=str,
		help="path to output hashes dictionary")
	ap.add_argument("-p", "--procs", type=int, default=-1,
		help="# of processes to spin up")
	args = vars(ap.parse_args())


import multiprocessing
from multiprocessing import Pool, Process, Manager

class MTSample_1:
    
    def __init__(self) -> None:
        self.num_list = ["1","2","3","4"]
    
    def count(self, name):
        sum = 0
        for i in range(1,10000):
            print(name, sum)
            sum += i
        
        return sum
    
    def run(self):
        pool = multiprocessing.Pool(processes=4)
        pool.map(self.count, self.num_list)
        pool.close()
        pool.join()

def mt_test_1():
    mt_sample = MTSample_1()
    mt_sample.run()

class MTSample_2:
    # json 빠르게 읽고 쓰기
    def __init__(self) -> None:
        
        pass

    def load_json(self, load_path, file_name):
        start_time = time.time()
        with open(f'{load_path}/{file_name}', "rb") as json_file:
            json_data = json.load(json_file)
        end_time = time.time()
        print("load_json = ", end_time-start_time)
        # print(len(json_data["images"]))
        return json_data

    def load_text(self, load_path, file_name):
        start_time = time.time()
        
        file = open(f"{load_path}/{file_name}", "rb")
        text_data = file.read()
        file.close()
        my_json = text_data.decode('utf-8')
        data = json.loads(my_json)
        # print(data["images"])
        end_time = time.time()
        # print(text_data)
        print("load_text = ", end_time-start_time)
        
        return data
    
    def save_json(self, json_data, save_path, file_name):
        start_time = time.time()
        # dict_to_byte = json.dumps(json_data).encode('utf-8')
        with open(f"{save_path}/{file_name}", "w") as json_file:
            json.dump(json_data, json_file)
            # json.dump(json_data, json_file, indent=4, separators=(',', ': '))
        end_time = time.time()
        print("save_json = ", end_time-start_time)
    
    def save_text(self, json_data, save_path, file_name):
        start_time = time.time()
        
        dict_to_byte = json.dumps(json_data).encode('utf-8')
        
        file = open(f"{save_path}/{file_name}", "wb")
        file.write(dict_to_byte)
        file.close()
        # print(data["images"])
        end_time = time.time()
        # print(text_data)
        print("save_text = ", end_time-start_time)
    
    def run(self):
        load_path = "./sample/json"
        save_path = "./sample/json/result"
        file_name = "data.json"
        
        json_data = self.load_json(load_path,file_name)
        text_data = self.load_text(load_path,file_name)
        
        self.save_json(json_data, save_path, f"json_{file_name}")
        self.save_text(text_data, save_path, f"text_{file_name}")
        
        result_json_data = self.load_json(save_path, f"json_{file_name}")
        result_text_data = self.load_text(save_path, f"text_{file_name}")
        
        print(len(result_json_data["images"]), len(result_json_data["categories"]), len(result_json_data["annotations"]))
        print(len(result_text_data["images"]), len(result_text_data["categories"]), len(result_text_data["annotations"]))
        

class MTSample_3(MTSample_2):
    def __init__(self) -> None:
        pass
        
    def make_dir(self, file_path, options={"is_remove": False}):
        is_remove = options["is_remove"]

        try:
            if is_remove:
                shutil.rmtree(file_path)
        except Exception as ex:
            pass
        try:
            os.makedirs(file_path)
        except Exception as ex:
            pass

    def load_file_list(self, file_path):
        file_names = os.listdir(file_path)
        return file_names
    
    def file_copy(self, dict_value):
        load_path = dict_value["load_path"]
        save_path = dict_value["save_path"]
        file_list = dict_value["file_list"]
        
        for file in file_list:
            src = f"{load_path}/{file}"
            dst = f"{save_path}/{file}"
            shutil.copy(src,dst)
    
    def single(self):
        load_file_path = "./sample/image"
        save_file_path = "./sample/copy_image"
        
        self.make_dir(save_file_path, options={"is_remove":True})
        start_time = time.time()
        
        file_list = self.load_file_list(load_file_path)
        dict_value = {
            "load_path":load_file_path,
            "save_path":save_file_path,
            "file_list":file_list
        }
        
        self.file_copy(dict_value)
        end_time = time.time()
        print(end_time-start_time)
        
    
    def run(self):
        # self.single()
        self.multi()
        
    def multi(self):
        load_file_path = "./sample/image"
        save_file_path = "./sample/copy_image"
        
        self.make_dir(save_file_path, options={"is_remove":True})
        start_time = time.time()
        
        file_list = self.load_file_list(load_file_path)
        cpu_count = multiprocessing.cpu_count()//2
        pool = multiprocessing.Pool(processes=cpu_count)
                
        dict_value = {
            "load_path":load_file_path,
            "save_path":save_file_path,
            "file_list":file_list[0:800]
        }
        pool.starmap(func=self.file_copy, iterable=[(dict_value,)])
        dict_value = {
            "load_path":load_file_path,
            "save_path":save_file_path,
            "file_list":file_list[800:-1]
        }
        pool.starmap(func=self.file_copy, iterable=[(dict_value,)])
        
        pool.close()
        pool.join()
        
        end_time = time.time()
        print(end_time-start_time)
        
def mt_test_3():
    mt_sample = MTSample_3()
    mt_sample.run()
    
import multiprocessing.sharedctypes as ms
import ctypes 
from multiprocessing import Lock
from multiprocessing.managers import BaseManager, SyncManager
from multiprocessing import Manager

class MathsClass:
    
    def add(self, step):
        result = 0
        for i in range(step):
            result += i
        print(result)
        return result
    
    def minus(self, step):
        result = 0
        for i in range(step):
            result -= i
        print(result)
        return result

from multiprocessing import Value, Array, Manager
class MyManager(SyncManager):
    pass

def mt_test_4():
    MyManager.register('Maths', MathsClass)
    
    with MyManager() as manager:
        maths = manager.Maths()
        # add = maths.add(1000)
        # minus = maths.minus(1000)
        num = manager.dict()
        arr = manager.list(range(10))
        
        p1 = Process(target=maths.add,args=(10000, ))
        p2 = Process(target=maths.minus,args=(10000, ))
        
        p1.start()
        p2.start()
        
        p1.join()
        p2.join()
    
    
def raw_read_1():
    import numpy as np
    from matplotlib import pyplot as plt
    
    file_path = "./sample/xray/example_9"
    file_name = "49_V_H_20220523-184900_580x413.raw"
    load_file_name = f"{file_path}/{file_name}"
    
    raw = open(load_file_name)
    #raw_new.tofile(raw)
    #raw.close()
    #raw.close()
    raw_image = np.fromfile(raw, dtype=np.uint16, sep="")
    
    raw_image = np.reshape(raw_image, [413,580]) 
    raw_image = raw_image%256
    
    options = {
        "file_name":f"{file_path}/save_{file_name}.png",
        "dtype": np.uint8,
        "start_pixel": 0,
        "end_pixel":255
    }
    save_image(raw_image, options)
    

def raw_read_2():
    file_path = "./sample/xray/example_8"
    file_name = "20191231113100_BSIU3027810_19DJSCE104I_TOP_HIGH.raw"
    load_file_name = f"{file_path}/{file_name}"
    
    raw = open(load_file_name)
    #raw_new.tofile(raw)
    #raw.close()
    #raw.close()
    fix_width = 5000
    
    raw_image = np.fromfile(raw, dtype=np.uint32, sep="")
    raw_image = np.reshape(raw_image, [len(raw_image)//fix_width, fix_width]) 
    data = raw_image.astype(np.int32)
    #잡음 문제 때문에 생기는 현상
    #1
    max_depth = np.iinfo(np.int32).max
    data[data==0] = np.max(data)
    data = (data - np.min(data))/(np.max(data)-np.min(data)) * np.iinfo(np.uint16).max
    data[data==np.max(data)] = 0
    #음영비 조절
    #2
    # data = 1.5*data-65535//2
    # data[data>65535] = 65535
    # data[data<0] = 0
    data = data.astype(np.uint16)
    
    #print(data.dtype)
    options = {
        "file_name":f"{file_path}/save_{file_name}.png",
        "dtype": np.uint16,
        "start_pixel": np.iinfo(data.dtype).min,
        "end_pixel": np.iinfo(data.dtype).max
    }
    save_image(data, options)
    
#########################################
from shutil import copyfile
import asyncio
from multiprocessing import Process, Pool, Queue, freeze_support
from library.utils.progress import Progress
import fnmatch
import glob

class Container:
    def __init__(self):
        pass
    
    def run(self, load_path, save_path):
        print(load_path)
    
        for (root, directories, files) in os.walk(load_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    
                    if file.endswith(".jpg"):
                        file_split_path = f"{save_path}/JPG/{file}"
                        copyfile(file_path,file_split_path)

                    if file.endswith(".png"):
                        file_split_path = f"{save_path}/PNG/{file}"
                        copyfile(file_path,file_split_path)

                    if file.endswith(".raw"):
                        file_split_path = f"{save_path}/RAW/{file}"
                        copyfile(file_path,file_split_path)

                    if file.endswith(".tif"):
                        file_split_path = f"{save_path}/TIF/{file}"
                        copyfile(file_path,file_split_path)

                    if file.endswith(".mpt"):
                        file_split_path = f"{save_path}/MTP/{file}"
                        copyfile(file_path,file_split_path)

                    if file.endswith(".txt"):
                        file_split_path = f"{save_path}/TXT/{file}"
                        copyfile(file_path,file_split_path)
                except:
                    pass

def container_run():
    container = Container()
    container.run()

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
    result = np.full((max_height, max_width, 3), 255)

    result[point_y:point_y + height, point_x:point_x + width] = data

    return result

def synthesis(data_a, data_b, options={}):
    '''
    options = {
        "alpha":{alpha},
        "beta":{beta},
        ""
    }
    '''
    alpha = options["alpha"]
    beta = options["beta"]
    
    max_pixel = 255
    
    alpha_data = max_pixel - alpha*data_a
    beta_data = max_pixel - beta*data_b

    result = alpha_data + beta_data
    return max_pixel-result


def container_mix_data():
    
    #1. background load
    load_path = "./sample/xray/example_10/background/"
    background_image_list = []
    for (root, directories, files) in os.walk(load_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                options = {
                    "file_name":file_path,
                    "dtype":cv2.IMREAD_COLOR
                }
                data = cv_load_image(options)
                background_image_list.append(data)
            except:
                pass

    #2. gun load
    load_path = "./sample/xray/example_10/object/crop/rifle_01220531/image"
    gun_image_list = []
    for (root, directories, files) in os.walk(load_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                options = {
                    "file_name":file_path,
                    "dtype":cv2.IMREAD_COLOR
                }
                data = cv_load_image(options)
                gun_image_list.append(data)
            except:
                pass
    
    #3. 합성
    for index, background in enumerate(background_image_list):
        
        random_gun_list = []
        for _ in range(4):
            random_gun_list.append(random.randrange(0,len(gun_image_list)))
            
        background_gun_image_list = []
        gun_option_list = []
        for gun_num in random_gun_list:
            #배경 고정
            gun = gun_image_list[gun_num]
            gun_height, gun_width = gun.shape[:2]
            height, width = background.shape[:2]
            gun_option = {
                "max_width": width,
                "max_height": height,
                "point_x": random.randrange(width//10*3,width//10*8),
                "point_y": random.randrange(height//10*2,height//10*4),
                "gun_width": gun_width,
                "gun_height": gun_height,
            }
            gun_option_list.append(gun_option)
            background_gun_image_list.append(background_fix_position(gun,gun_option))
            
        background_1 = copy.copy(background)
        for gun in background_gun_image_list:
            #합성
            synthesis_options = {
                "alpha":1,
                "beta":1
            }
            background_1 = synthesis(background_1,gun,synthesis_options)

        save_path = f"./sample/xray/example_10/result/image_{index}.jpg"
        save_option = {
            "file_name": save_path,
            "dtype": np.uint8,
            "start_pixel": 0,
            "end_pixel": 255
        }
        
        cv_save_image(background_1, save_option)
        # 사각형 치기
        for gun_option in gun_option_list:
            #print(gun_option)
            x = gun_option["point_x"]
            y = gun_option["point_y"]
            width = gun_option["gun_width"]+x
            height = gun_option["gun_height"]+y
            
            background_1 = cv2.rectangle(background_1, (x,y), (width,height),(0,0,255),3 )
        
        
        save_path = f"./sample/xray/example_10/result/image_rect_{index}.jpg"
        save_option = {
            "file_name": save_path,
            "dtype": np.uint8,
            "start_pixel": 0,
            "end_pixel": 255
        }
        cv_save_image(background_1, save_option)
        pass

def crop_test():
    
    load_path = "./sample/xray/example_10/object/crop/rifle_00220531/json"
    file_name = "data.json"
    json_data = load_json(load_path, file_name)
    
    annotations = json_data["annotations"]
    images = json_data["images"]
    categories = json_data["categories"]
    
    for annotation in annotations:
        path = images[annotation["image_id"]]["path"]
        load_option = {
            "file_name": path,
            "dtype":cv2.IMREAD_GRAYSCALE
        }
    
        data = cv_load_image(load_option)
        segmentation = seg_to_list(annotation["segmentation"][0])
        result = mask_polygon(data,segmentation)
        
        save_option = {
            "file_name": path,
            "dtype": np.uint8,
            "start_pixel": 0,
            "end_pixel": 255
        }
        cv_save_image(result, save_option)
        
    '''
    segmentation = json_data["annotations"][0]["segmentation"][0]
    x,y,width,height = json_data["annotations"][0]["bbox"]
    
    result = []
    for i in range(0,len(segmentation),2):
        result.append([segmentation[i],segmentation[i+1]])
    
    result = crop_mask(gun_image_result_list[0],result)

    save_path = f"./sample/xray/example_10/result/test.png"
    save_option = {
        "file_name": save_path,
        "dtype": np.uint8,
        "start_pixel": 0,
        "end_pixel": 255
    }
    
    cv_save_image(result, save_option)
    '''
    
def seg_to_list(segmentation):
    result = []
    for i in range(0,len(segmentation),2):
        result.append([segmentation[i],segmentation[i+1]])
        
    return result

def mask_polygon(data, segmentation):
    result = copy.copy(data)
    if len(result.shape) == 3:
        result = result[:,:,0]
    height, width = result.shape[:2]
    
    mask = np.full( (height, width), 255, dtype = np.int32)
    segmentation = np.array(segmentation, dtype = np.int32)

    cv2.fillPoly(mask, [segmentation], 1)
    result = result * mask
    #result[result==0]=np.iinfo(result.dtype).max
    return result

def image_24_to_16():
    #1. background load
    
    #load_path = "./sample/xray/example_10/convert/group/image"
    #load_path = "./sample/xray/example_11/origin/"
    
    # load_path = "E://container/신항2센터/JPG/SIDE"
    # save_path = "E://container/신항2센터/JPG/16bit/SIDE"

    load_path = "./sample/xray/example_12/4types/image"
    save_path = "./sample/xray/example_12/4types/16bit"
    
    background_image_list = []
    background_name_list = []
    
    for (root, directories, files) in os.walk(load_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                #background_name_list.append(file)
                options = {
                    "file_name":file_path,
                    "dtype":-1
                }
                #print(options)
                data = cv_load_image(options)
                #print(np.min(data), np.max(data))
                #background_image_list.append(data)
                name = save_path+"/"+file[:-3]+"png"
                #print(name)
                data = data * 65535
                data = data.astype(np.int32)
                
                save_option = {
                    "file_name": name,
                    "dtype": np.uint16,
                    "start_pixel": 0,
                    "end_pixel": 65535
                }
                save_image(data, save_option)
                
            except Exception as ex:
                #print(ex)
                pass


def xray_excel():
    import pandas as pd
    load_path = "./sample/excel/example_1/excel.xlsx"
    df = pd.read_excel(load_path)
    data = df.to_numpy()
    #print(data)
    result = (data-np.min(data))/(np.max(data)-np.min(data))
    
    data = result * 65535
    
    print(np.shape(data))
    print(np.min(data), np.max(data))
    
    data = data.astype(np.int32)

    save_name =  "./sample/excel/example_1/excel.png"
    save_option = {
        "file_name": save_name,
        "dtype": np.uint16,
        "start_pixel": 0,
        "end_pixel": 65535
    }
    save_image(data, save_option)
    
    pass


def split_dataset():
    name = "test"
    image_path = "E:/sample/xray/example_16/ETRI/origin/image"
    save_path = f"E:/sample/xray/example_16/ETRI/origin/{name}/image"
    
    make_dir(save_path)
    
    load_path = "E:/sample/xray/example_16/ETRI/origin/json"
    file_name = f"{name}.json"
    json_data = load_json(load_path, file_name)
    images = json_data["images"]
    
    for image in images:
        try:
            origin_image_file = f"{image_path}/{image['file_name']}"
            copy_image_file = f"{save_path}/{image['file_name']}"
            shutil.copy(origin_image_file,copy_image_file)
        except Exception as e:
            pass
    pass

def split_top_side():
    name = "train"
    image_path = f"E:/sample/xray/example_16/ETRI/split/{name}/origin/image"

    save_top_path = f"E:/sample/xray/example_16/ETRI/split/{name}/TOP_PNG/image"
    make_dir(save_top_path)

    save_side_path = f"E:/sample/xray/example_16/ETRI/split/{name}/SIDE_PNG/image"
    make_dir(save_side_path)
    
    load_path = "E:/sample/xray/example_16/ETRI/origin/json"
    file_name = f"{name}.json"
    json_data = load_json(load_path, file_name)
    
    info = json_data["info"]
    images = json_data["images"]
    categories = json_data["categories"]
    annotations = json_data["annotations"]
    
    top_json = {
        "info":info,
        "images":[],
        "categories":categories,
        "annotations":[]
    }
    side_json = {
        "info":info,
        "images":[],
        "categories":categories,
        "annotations":[]
    }
    
    for image in images:
        origin_image_file = f"{image_path}/{image['file_name']}"
        copy_image = copy.copy(image)
        if image["scan_mode"] == "SIDE":
            copy_image_file = f"{save_side_path}/{image['file_name'][:-4]}.png"
            copy_image["file_name"] = f"{image['file_name'][:-4]}.png"
            side_json["images"].append(copy_image)
        else:
            copy_image_file = f"{save_top_path}/{image['file_name'][:-4]}.png"
            copy_image["file_name"] = f"{image['file_name'][:-4]}.png"
            top_json["images"].append(copy_image)
        
        #shutil.copy(origin_image_file,copy_image_file)
        data = load_image(options={
            "file_name":origin_image_file,
            "dtype":np.uint16
        })
        save_image(data, options={
            "file_name":copy_image_file,
            "dtype":np.uint16,
            "start_pixel":0,
            "end_pixel":65535
        })
        for ann_idx, annotation in enumerate(annotations):
            if annotation["image_id"] == image["id"]:
                copy_annotation = copy.copy(annotation)
            
                if image["scan_mode"] == "SIDE":
                    side_json["annotations"].append(copy_annotation)
                else: #Top
                    top_json["annotations"].append(copy_annotation)

                del annotations[ann_idx]
    
    save_top_json_path = f"E:/sample/xray/example_16/ETRI/split/{name}/TOP_PNG/json"
    make_dir(save_top_json_path)

    save_side_json_path = f"E:/sample/xray/example_16/ETRI/split/{name}/SIDE_PNG/json"
    make_dir(save_top_json_path)

    save_json(top_json,save_top_json_path,"data.json")
    save_json(side_json,save_side_json_path,"data.json")


def test4():
    annotations = [
        {"image_id":1, "data":1},
        {"image_id":2, "data":1},
        {"image_id":3, "data":1},
        {"image_id":4, "data":1}
    ]
    
    data = [annotation["image_id"] for annotation in annotations]    
    print(data)
    pass

if __name__ == '__main__':
    # cv_imread_16bit_3channel()
    # opencv_remap_test()
    # opencv_affine_test()
    # opencv_rotate_test()
    # opencv_histogram_test()

    # spectrum_map()
    # spectrum_run()

    # cv_imread_color_test()
    # mt_test_1()
    # mt_test_2()
    # mt_test_3()
    # mt_test_4()
    
    # raw_read_1()
    # raw_read_2()
    # container_run()
    # container_mix_data()
    # crop_test()
    # image_24_to_16()
    # xray_excel()
    # test3()
    # split_dataset()
    # test_del()
    split_top_side()
    # test4()
    pass