from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

from library.images.rotate import *

@dec_func_start_end
def rotate_image_annotation():

    coco = Rotate()
    names = ["data"]

    # origin_path = "F://custom/_seperation_/reside_remove"
    # save_path = "F://custom/_seperation_/rotate_remove"
    
    origin_path = "F://ct/_train_/origin"
    save_path = "F://ct/_train_/origin_rotate"

    # origin_path = "E://daq/_train_/crop"
    # save_path = "E://daq/_train_/rotate"

    # origin_path = "E://ct/_train_/rate"
    # save_path = "E://ct/_train_/rotate"

    make_dir(save_path,{"is_remove":True})

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
                "angle_split": 45,
            }
            coco.run(options)
            
            ground_truth_options = {
                "image_path": f"{save_path}/{objects_name}/image",
                "json_path": f"{save_path}/{objects_name}/json/{name}.json",
                "gt_path": f"{save_path}/{objects_name}/test"
            }
            ground_truth_view(ground_truth_options)
    

class Rotate(Bone):

    def run(self, options):
        '''
        options = {
            "image_path":{image_path},
            "json_path":{json_path},
            "save_image_path":{save_image_path},
            "save_json_path":{save_json_path},
            "angle_split":{angle_split}
        }
        '''
        try:
            image_path = options["image_path"]
            json_path = options["json_path"]

            save_image_path = options["save_image_path"]
            save_json_path = options["save_json_path"]

            angle_split = options["angle_split"]

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

            image_idx = 0
            annotation_idx = 0

            json_result = {
                "images":[],
                "categories":categories,
                "annotations":[],
            }

            for image in images:
                image_id = image["id"]
                image_path, image_file = self.split_path_name(image["path"])
                
                for angle in range(0, 360, angle_split):
                    data = self.cv_load_image(image_path, image_file)
                    rotate_options = {
                        "angle": angle
                    }
                    rotate_data, rotate_matrix = rotate(data, rotate_options)
                    
                    for annotation in annotations:
                        copy_annotation = copy.copy(annotation)

                        annotation_image_id = copy_annotation["image_id"]
                        if image_id == annotation_image_id:
                            segmentation = copy_annotation['segmentation']
                            rotate_segmentation = self._segmentation(rotate_matrix, segmentation)
                            rotate_box = self._bbox(rotate_segmentation)
                            rotate_data = self.mask_filter_background(rotate_data, rotate_segmentation)
                            
                            copy_annotation["segmentation"] = rotate_segmentation
                            copy_annotation["bbox"] = rotate_box
                            copy_annotation["id"] = annotation_idx
                            copy_annotation["image_id"] = image_idx
                            annotation_idx += 1
                            json_result["annotations"].append(copy_annotation)
                    
                    copy_image = copy.copy(image)
                
                    copy_image["id"] = image_idx
                    copy_image["width"],copy_image["height"] = np.shape(rotate_data)
                    copy_image["path"] = f"{save_image_path}/{annotation_idx}.png"
                    copy_image["file_name"] = f"{annotation_idx}.png"
                    
                    self.save_image(rotate_data,save_image_path,copy_image["file_name"])
                    
                    json_result["images"].append(copy_image)
                    image_idx += 1
            
            save_options = {
                "load_path":save_json_path,
                "file_name":json_file
            }
            self.save_json(json_result,save_options)
        except Exception as ex:
            logger_exception(ex)
            
    def _bbox(self,segmentation):
        new_seg_even = segmentation[0][::2]
        new_seg_odd = segmentation[0][1::2]

        min_x = min(new_seg_even)
        min_y = min(new_seg_odd)
        max_x = round(max(new_seg_even) - min_x, 1)
        max_y = round(max(new_seg_odd) - min_y, 1)
        result = [min_x, min_y, max_x, max_y]
        
        return result

    def _segmentation(self, rotation_matrix, segmentation):
        result = []
        for seg in segmentation:
            seg_array = np.array(seg)
            x = seg_array[::2]
            y = seg_array[1::2]
            one = np.ones_like(x)
            x_y = np.array([x, y, one])
            new_x_y = rotation_matrix @ x_y
            new_x = new_x_y[0, :]
            new_y = new_x_y[1, :]
            seg_array[::2] = new_x
            seg_array[1::2] = new_y
            seg_array = np.round_(seg_array, 1)  # 소수점 1자리수에서 반올림
            result.append(seg_array.tolist())
        return result

    def mask_filter_background(self, data, segmentation):
        
        height, width = np.shape(data)
        mask = np.full((height,width),0)
        
        segmentation = segmentation[0]
        len_segmentation = (len(segmentation))//2
        segmentation = np.array(segmentation, dtype=np.int32).reshape((len_segmentation,2))
        cv2.fillPoly(mask, [segmentation], 1, cv2.LINE_AA)
        
        mask_16 = np.array(mask, dtype=np.uint16)
        #mask_16[mask_16==1] = 65535
        
        result = np.full((height,width),65535, dtype=np.uint16)
        result = np.where(
            mask_16 == 1,
            data,
            result
        )
        result[result==0] = 65535
        return result