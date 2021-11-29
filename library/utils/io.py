from library.utils.header import *
from library.utils.decorator import *

def load_image(options):
    '''
    options = {
        "file_name": {file_name},
        "dtype": {dtype}
    }
    '''
    file_name = options["file_name"]
    dtype = options["dtype"]

    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype=dtype)
    return data

def save_image(data, options):
    '''
    options = {
        "file_name": {file_name},
        "dtype": {dtype},
        "start_pixel": {start_pixel},
        "end_pixel": {end_pixel}
    }
    '''
    file_name = options["file_name"]
    dtype = options["dtype"]
    start_pixel = options["start_pixel"]
    end_pixel = options["end_pixel"]

    img = Image.fromarray(
        np.asarray(
            np.clip(data, start_pixel, end_pixel), dtype=dtype
        )
    )
    img.save(file_name)

def save_raw(data, options):
    '''
    options = {
        "file_name": {file_name},
    }
    '''
    file_name = options["file_name"]

    data = data.flatten()
    myfmt = 'f' * len(data)
    file = open(file_name, "wb")
    binary = struct.pack(myfmt, *data)
    file.write(binary)
    file.close()

def cv_load_image(options):
    '''
    options = {
        "file_name": {file_name},
        "dtype": {dtype},
    }
    '''
    file_name = options["file_name"]
    dtype = options["dtype"]
    
    image_array = np.fromfile(file_name, np.uint8)
    result = cv2.imdecode(image_array, dtype)
    return result

def cv_save_image(data, options):
    '''
    options = {
        "file_name": {file_name},
    }
    '''
    file_name = options["file_name"]
    dtype = options["dtype"]
    start_pixel = options["start_pixel"]
    end_pixel = options["end_pixel"]

    extension = os.path.splitext(file_name)[1]
    data = np.asarray(
        np.clip(data, start_pixel, end_pixel), dtype=dtype
    )
    # print(data.dtype)
    result, encoded_img = cv2.imencode(extension, data)
    # cv2.imwrite(path_name,data)
    if result: 
        with open(file_name, mode="w+b") as f: 
            encoded_img.tofile(f)
    
def load_dir_list(options):
    '''
    options = {
        "dir_path": {dir_path},
    }
    '''
    dir_path = options["dir_path"]
    folders = os.listdir(dir_path)
    
    full_result = []
    path_result = []
    name_result = []

    for folder in folders:
        path_result.append(dir_path)
        name_result.append(folder)
        full_result.append(f"{dir_path}/{folder}")

    return full_result,path_result,name_result

def delete_dir_list(options):
    '''
    options = {
        "dir_path": {dir_path},
    }
    '''
    dir_path = options["dir_path"]
    dir_list = os.listdir(dir_path)
    for dir_name in dir_list:
        os.remove(f"{dir_path}/{dir_name}")

def load_file_list(options):
    '''
    options = {
        "ends_with": {ends_with},
        "file_path": {file_path}
    }
    '''
    ends_with = options["ends_with"]
    file_path = options["file_path"]
    
    full_result = []
    path_result = []
    name_result = []

    file_names = os.listdir(file_path)
    
    for file_name in file_names:
        if file_name.endswith(ends_with):
            full_file_name = os.path.join(file_path, file_name)
            full_file_name = full_file_name.replace("\\","/")
            full_result.append(full_file_name)
            path_result.append(file_path)
            name_result.append(file_name)

    return full_result, path_result, name_result

def change_file_list_name(options):
    '''
    options = {
        "ends_with": {ends_with},
        "file_path": {file_path},
        "save_path": {save_path}
    }
    '''
    ends_with = options["ends_with"]
    file_path = options["file_path"]
    save_path = options["save_path"]

    rename_key = 0

    file_list_options = {
        "ends_with":ends_with,
        "file_path":file_path
    }

    fulls, _, _ = load_file_list(file_list_options)
    index = rename_key

    for full in fulls:
        try:
            save = f"{save_path}/{index}{ends_with}"
            copyfile(full, save)
        except Exception as ex:
            pass
        index += 1

def copy_dir_list(options):
    '''
    options = {
        "dir_path": {dir_path},
        "save_path": {save_path}
    }
    '''
    dir_path = options["dir_path"]
    save_path = options["save_path"]
    # shutil.copy(dir_path, save_path)
    copy_tree(dir_path, save_path)

def make_dir(file_path, options={"is_remove": False}):
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


def load_json(load_path, file_name):
    with open(f'{load_path}/{file_name}') as json_file:
        json_data = json.load(json_file)
    return json_data


def save_json(json_data, save_path, file_name):
    with open(f"{save_path}/{file_name}", "w") as json_file:
        json.dump(json_data, json_file, indent=4, separators=(',', ': '))


def split_path_name(file_path_name):
    path = "/".join(file_path_name.split("/")[:-1])
    name = file_path_name.split("/")[-1]
    return path, name

def split_file_extension(file_name):
    name = file_name.split(".")[0]
    extension = file_name.split(".")[-1]
    return name, extension
