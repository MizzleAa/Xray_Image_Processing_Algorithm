from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

from library.images.resize import *

@dec_func_start_end
def rate_image_annotation():

    coco = Rate()
    names = ["data"]

    # origin_path = "E://daq/_train_/rotate"
    # save_path = "E://daq/_train_/rate"
    # origin_path = "E://ct/_train_/reside"
    # save_path = "E://ct/_train_/rate"
    
    origin_path = "F://ct/_train_/origin_rotate"
    save_path = "F://ct/_train_/rate"

    make_dir(save_path,options={"is_remove":True})

    dir_list_options = {
        "dir_path": origin_path
    }

    paths, _, _ = load_dir_list(dir_list_options)

    progress = Progress(max_num=len(paths),work_name=__name__)
    
    for path in paths:
        progress.set_work_name(f" = {path}\n")
        progress.update()

        for name in names:
            path_options = {
                "image_path": f"{path}/image",
                "json_path": f"{path}/json/{name}.json",
            }
            coco.change_image_path(path_options)

            objects_name = path.split("/")[-1]
            options = {
                "image_path": f"{path}/image",
                "json_path": f"{path}/json/{name}.json",
                "save_image_path": f"{save_path}/{objects_name}/image",
                "save_json_path": f"{save_path}/{objects_name}/json",
                "rate_width": 0.20, #1.0118577
                "rate_height": 0.20, #1.0118577
            }
            coco.run(options)

            ground_truth_options = {
                "image_path": f"{save_path}/{objects_name}/image",
                "json_path": f"{save_path}/{objects_name}/json/{name}.json",
                "gt_path": f"{save_path}/{objects_name}/test"
            }
            ground_truth_view(ground_truth_options)


class Rate(Bone):

    def run(self, options):
        '''
        options = {
            "image_path":{image_path},
            "json_path":{json_path},
            "save_image_path":{save_image_path},
            "save_json_path":{save_json_path},
            "rate_width":{rate_width},
            "rate_height":{rate_height}
        }
        '''
        try:
            image_path = options["image_path"]
            json_path = options["json_path"]

            save_image_path = options["save_image_path"]
            save_json_path = options["save_json_path"]
            
            rate_width = options["rate_width"]
            rate_height = options["rate_height"]

            make_dir(save_image_path)
            make_dir(save_json_path)

            json_path, json_file = self.split_path_name(json_path)

            json_data_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            json_data = self.load_json(json_data_options)

            images = self.get_images(json_data)
            annotations = self.get_annotations(json_data)
            categories = self.get_categories(json_data)

            file_list_options = {
                "ends_with":"png",
                "file_path":image_path
            }
            _,_,files = load_file_list(file_list_options)

            resize_options = {
                "width":rate_width,
                "height":rate_height,
            }

            for file in files:
                data  = self.load_image(image_path,file)
                height, width = np.shape(data)

                resize_options = {
                    "width":int(width*rate_width),
                    "height":int(height*rate_height),
                }

                resize_data = resize(data,resize_options)
                self.save_image(resize_data,save_image_path,file)
                
            
            json_result = {
                "images":[],
                "categories":categories,
                "annotations":[],
            }

            image_idx = 0
            annotation_idx = 0

            for image in images:
                copy_image = copy.copy(image)

                image_id = copy_image["id"]
                image_width = copy_image["width"]
                image_height = copy_image["height"]
                

                for annotation in annotations:
                    
                    annotation_image_id = annotation["image_id"]
                    if image_id == annotation_image_id:
                        segmentation = annotation["segmentation"]
                        bbox = annotation["bbox"]

                        width = int(rate_width*image_width)
                        height = int(rate_height*image_height)
                        
                        copy_annotation = copy.copy(annotation)
                        copy_annotation["id"] = annotation_idx
                        copy_annotation["image_id"] = image_idx
                        
                        copy_annotation["segmentation"] = self._segmentation(segmentation,rate_width,rate_height)
                        copy_annotation["bbox"] = self._bbox(bbox,rate_width,rate_height)

                        json_result["annotations"].append(copy_annotation)
                        annotation_idx += 1

                copy_image["id"] = image_idx
                copy_image["width"] = rate_width
                copy_image["height"] = rate_height
                
                json_result["images"].append(copy_image)

                image_idx += 1


            save_options = {
                "load_path":save_json_path,
                "file_name":json_file
            }

            self.save_json(json_result, save_options)
        except Exception as ex:
            logger_exception(ex)
            

    def _bbox(self, bbox, width, height):
        return [bbox[0]*width,bbox[1]*height,bbox[2]*width,bbox[3]*height]

    def _segmentation(self, segmentation, width, height):
        result = []
        for seg in segmentation:
            seg_array = np.array(seg)
            seg_array[::2] *= width
            seg_array[1::2] *= height
            seg_array = np.round_(seg_array, 1)  # 소수점 1자리수에서 반올림
            result.append(seg_array.tolist())
        return result
