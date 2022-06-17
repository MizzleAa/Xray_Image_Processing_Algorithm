from library.utils.header import *
from library.utils.io import *
from library.utils.view import *

class Container:
    def __init__(self):
        pass
    
    def run(self, load_path, save_path):
        print(load_path)
    
        for (root, directories, files) in os.walk(load_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    print(file_path)
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
    
    def split(self, load_path, min_size):
        for (root, directories, files) in os.walk(load_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) 
                if file_size > 100000:
                    try:
                        if "SIDE" in file:
                            print(f"SIDE = {file}, {file_size}")
                            file_split_path = f"{load_path}/SIDE/{file}"
                            copyfile(file_path,file_split_path)
                        if "TOP" in file:
                            print(f"TOP = {file}, {file_size}")
                            file_split_path = f"{load_path}/TOP/{file}"
                            copyfile(file_path,file_split_path)
                    except:
                        pass
    
    def slice(self, load_path, min_size, split_line):
        for (root, directories, files) in os.walk(load_path):
            print(root)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) 
                if file_size > 1000000:
                    print(f"{file_path}, {file_size}")
                    load_options = {
                        "file_name": file_path,
                        "dtype": np.uint8
                    }
                    image = load_image(load_options)
                    top = image[:split_line,:]
                    side = image[split_line:,:]
                    #######################################
                    
                    #cv_view(side)
                    file_split_path = f"{load_path}/SIDE/{file}"
                    save_side_options = {
                        "file_name":file_split_path,
                        "dtype":np.uint8,
                        "start_pixel":np.iinfo(side.dtype).min,
                        "end_pixel":np.iinfo(side.dtype).max
                    }
                    cv_save_image(side, save_side_options)
                    #######################################
                    file_split_path = f"{load_path}/TOP/{file}"
                    save_top_options = {
                        "file_name":file_split_path,
                        "dtype":np.uint8,
                        "start_pixel":np.iinfo(top.dtype).min,
                        "end_pixel":np.iinfo(top.dtype).max
                    }
                    save_image(top, save_top_options)
                    break
            break  
        pass
    
def container_run():
    container = Container()
    container.run("G://신선대","E://container/신선대")
    container.run("G://신항1센터","E://container/신항1센터")
    container.run("G://신항2센터","E://container/신항2센터")


def container_split():
    container = Container()
    #container.split("E://container/신선대/JPG", min_size=100000)
    #container.split("E://container/신항1센터/JPG", min_size=100000)
    container.slice("E://container/신항2센터/JPG", min_size=100000, split_line=1135)
    
    pass
    

if __name__ == '__main__':
    #container_run()
    container_split()
    pass