from library.utils.header import *
from library.utils.io import *

class Bone:
    def __init__(self):
        pass
    
    def load_json(self, options):
        '''
        options = {
            "load_path": {"load_path"},
            "file_name": {"file_name"},
        }
        '''
        load_path = options["load_path"]
        file_name = options["file_name"]
        json_data = load_json(load_path,file_name)
        return json_data

    def save_json(self, json_data, options):
        '''
        options = {
            "load_path": {"load_path"},
            "file_name": {"file_name"},
        }
        '''
        save_path = options["load_path"]
        file_name = options["file_name"]

        save_json(json_data,save_path,file_name)
        
    def init_annotation_dataset(self):
        result = {
            "id": 0,
            "image_id": 0,
            "category_id": 0,
            "segmentation": [[]],
            "area": 2,
            "bbox": [],
            "iscrowd": False,
            "isbbox": False,
            "color": "#000000",
            "metadata": {}
        }
        return result

    def init_category_dataset(self):
        result = {
            "id": 0,
            "name": "",
            "supercategory": "",
            "color": "#000000",
            "metadata": "",
            "keypoint_colors": ""
        }
        return result

    def init_image_dataset(self):
        result = {
            "id": 0,
            "dataset_id": 0,
            "category_ids": [],
            "path": "",
            "width": 0,
            "height": 0,
            "file_name": "",
            "annotated": False,
            "annotating": [],
            "num_annotations": 0,
            "metadata": {},
            "deleted": False,
            "milliseconds": 0,
            "events": [],
            "regenerate_thumbnail": False,
        }
        return result

    def get_annotations(self, json_data):
        return copy.copy(json_data["annotations"])

    def get_images(self, json_data):
        return copy.copy(json_data["images"])

    def get_categories(self, json_data):
        return copy.copy(json_data["categories"])

    def get_split(self, json_data):
        images = self.get_images(json_data)
        categoires = self.get_categories(json_data)
        annotations = self.get_annotations(json_data)

        return images, categoires, annotations

    def to_list_key_parameter(self, json_data, options):
        key = options["key"]
        select = options["select"]
        parameters = []

        if select == "images":
            parameters = self.get_images(json_data)
        if select == "annotations":
            parameters = self.get_annotations(json_data)
        if select == "categories":
            parameters = self.get_categories(json_data)

        result = []
        for parameter in parameters:
            result.append(parameter[str(key)])

        return result

    def split_path_name(self, file_path_name):
        path = "/".join(file_path_name.split("/")[:-1])
        name = file_path_name.split("/")[-1]
        return path, name

    def change_image_path(self, options):
        try:
            image_path = options["image_path"]
            json_path = options["json_path"]
            ends_with = "png"

            json_path, json_file = self.split_path_name(json_path)

            json_data_options = {
                "load_path":json_path,
                "file_name":json_file
            }

            json_data = self.load_json(json_data_options)
            
            images = self.get_images(json_data)
            annotations = self.get_annotations(json_data)
            categories = self.get_categories(json_data)

            json_result = {
                "images":[],
                "categories":categories,
                "annotations":annotations,
            }

            for image in images:
                file_name = f"{image['file_name'].split('.')[0]}.{ends_with}"

                copy_image = copy.copy(image)
                copy_image["path"] = f"{image_path}/{file_name}"
                copy_image["file_name"] = file_name

                json_result["images"].append(copy_image)

            save_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            
            self.save_json(json_result, save_options)
        except Exception as ex:
            logger_exception(ex)
        
    def inner_point(self, background, inner, options):
        padding_width = options["padding_width"]
        padding_height = options["padding_height"]
        
        background_height, background_width = np.shape(background)
        inner_height, inner_width = np.shape(inner)

        height = background_height - inner_height
        width = background_width - inner_width

        result_width = random.randrange(padding_width, width-padding_width)
        result_height = random.randrange(padding_height, height-padding_height)

        return result_width, result_height

    def load_image(self, load_path, file_name):
        load_file_path_name = f"{load_path}/{file_name}"

        load_image_options = {
            "file_name": load_file_path_name,
            "dtype": "int32"
        }
        data = load_image(load_image_options)
        return data

    def cv_load_image(self, load_path, file_name):
        load_file_path_name = f"{load_path}/{file_name}"

        load_image_options = {
            "file_name": load_file_path_name,
            "dtype": cv2.IMREAD_UNCHANGED 
        }
        data = cv_load_image(load_image_options)
        return data
        
    def save_image(self, data, save_path, file_name, options={"dtype": "uint16", "end_pixel": 65535}):
        save_file_path_name = f"{save_path}/{file_name}"

        save_image_options = {
            "file_name": save_file_path_name,
            "dtype": options["dtype"],
            "start_pixel": 0,
            "end_pixel": options["end_pixel"]
        }
        # data = data/np.max(data)*options["end_pixel"]
        # save_image(data, save_image_options)
        cv_save_image(data,save_image_options)
        
        return data

    def is_ben(self, data, ben_list):
        check = False
        for ben in ben_list:
            if data == ben:
                check = True
                break

        return check
    
    def information_json(self, json_data):
        images = self.get_images(json_data)
        annotations = self.get_annotations(json_data)
        categories = self.get_categories(json_data)

        return images, annotations, categories
        