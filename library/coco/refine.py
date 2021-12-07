from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

from library.images.synthesis import *
from library.images.backgroundfix import *
from library.images.segmentation import *

@dec_func_start_end
def make_auto_refine_image_annotation():

    coco = AutoRefine()
    
    path = "F:/custom/_others_/origin"
    

    dir_list_options = {
        "dir_path": path
    }
    
    global_fulls, global_paths, global_names = load_dir_list(dir_list_options)

    progress = Progress(max_num=len(global_fulls),work_name=__name__)

    for full, path, name in zip(global_fulls, global_paths, global_names):
        progress.set_work_name(f" = {full}\n")
        progress.update()

        result = "auto_refine"
        
        options = {
            "ends_with": ".png",
            "file_path": f"{full}/image",
            "save_path": f"{full}/refine"
        }
        # print(name)
        change_file_list_name(options)
        category_name = name
        # category_name = "null"
        auto_refine_image_category_annotation_options = {
            "image_path": f"{full}/refine",
            "ends_with": ".png",

            "save_image_path": f"{full}/{result}_image",
            "save_json_path": f"{full}/{result}_json/",

            "category_name": category_name,
            "category_id": 0,

            "segmentation_options": {
                "min_area": 1000,
                "cliping_dtype": "uint16",
                "threshold": 60000,
                "max_pixel": 1
            }
            
        }
        coco.run(auto_refine_image_category_annotation_options)

        ground_truth_options = {
            "image_path": f"{full}/{result}_image",
            "json_path": f"{full}/{result}_json/data.json",
            "gt_path": f"{full}/test"
        }
        ground_truth_view(ground_truth_options)
        
        
class AutoRefine(Bone):

    def run(self, options):
        image_path = options["image_path"]
        ends_with = options["ends_with"]
        
        save_image_path = options["save_image_path"]
        save_json_path = options["save_json_path"]

        category_name = options["category_name"]
        category_id = options["category_id"]

        segmentation_options = options["segmentation_options"]

        annotation_id_count = 0
        image_id_count = 0

        json_result = {
            "images": [],
            "categories": [],
            "annotations": [],
        }

        category = self.init_category_dataset()

        category["id"] = category_id
        category["name"] = category_name
        
        json_result["categories"].append(category)

        make_dir(save_image_path,options={"is_remove": True})
        make_dir(save_json_path,options={"is_remove": True})

        file_list_options = {
            "file_path":image_path,
            "ends_with":ends_with
        }
        
        _, paths, files = load_file_list(file_list_options)

        for path, file in zip(paths,files):
            data = self.load_image(path,file)
            
            image = self.init_image_dataset()
            image["id"] = image_id_count
            image["path"] = path
            image["file_name"] = file
            image["height"],image["width"] = np.shape(data)

            self.save_image(data,save_image_path,file)

            contours, _, _ = segmentation(data, segmentation_options)
            
            for contour in contours:
                annotation = self.init_annotation_dataset()
                annotation["bbox"] = self._bbox(contour)
                annotation["segmentation"] = self._segmentation(contour)
                annotation["category_id"] = category_id
                annotation["id"] = annotation_id_count
                annotation["image_id"] = image_id_count
                
                annotation_id_count += 1
                json_result["annotations"].append(annotation)

            image_id_count += 1
            json_result["images"].append(image)

            save_options = {
                "load_path":save_json_path,
                "file_name":"data.json"
            }
            self.save_json(json_result, save_options)
    
    def _bbox(self, contour):
        x,y,z = np.shape(contour)
        data = np.reshape(contour,(x,z)).T
        x = int(np.min(data[0]))
        y = int(np.min(data[1]))
        width = int(np.max(data[0]) - x)
        height = int(np.max(data[1]) - y)
        
        return [x,y,width,height]

    def _segmentation(self, contour):
        x,y,z = np.shape(contour)
        size = x*y*z
        annotation_segmentations = [np.reshape(contour,size).tolist()]
        return annotation_segmentations