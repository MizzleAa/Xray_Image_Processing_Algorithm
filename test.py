from library.utils.header import *

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
    
    data = np.zeros(shape=(height,width,3))
    
    for h in range(height):
        for w in range(width):
            l_value = low[h][w]
            h_value = high[h][w]

            l_value = l_value // 256 - 1
            h_value = h_value // 256 - 1
            data[h][w] = s_map[l_value][h_value]

    cv2.imshow('spectrum_run', data)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
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
    pass