from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

from library.images.synthesis import *
from library.images.backgroundfix import *

    

@dec_func_start_end
def single_synthesis_image_annotation():

    coco = SingleSynthesis()
    names = ["data"]

    origin_path = "E://daq/_train_/crop_finish"
    save_path = "D://daq/_train_/synthesis"
    background_path = "E://daq/_background_/null/background_fix"

    make_dir(save_path,{"is_remove":True})

    dir_list_options = {
        "dir_path": origin_path
    }

    fulls, _, _ = load_dir_list(dir_list_options)

    progress = Progress(max_num=len(fulls),work_name=__name__)
    
    for full in fulls:
        progress.set_work_name(f" = {full}\n")
        progress.update()

        for name in names:
            path_options = {
                "image_path": f"{full}/image",
                "json_path": f"{full}/json/{name}.json",
            }
            coco.change_image_path(path_options)

            objects_name = full.split("/")[-1]
            options = {
                "image_path": f"{full}/image",
                "json_path": f"{full}/json/{name}.json",
                "save_image_path": f"{save_path}/{objects_name}/image",
                "save_json_path": f"{save_path}/{objects_name}/json",
                "ends_with": ".png",
                "file_path": background_path,
                "inner_point_options": {
                    "padding_width":150,
                    "padding_height":150
                }
            }
            coco.run(options)
            
            ground_truth_options = {
                "image_path": f"{save_path}/{objects_name}/image",
                "json_path": f"{save_path}/{objects_name}/json/{name}.json",
                "gt_path": f"{save_path}/{objects_name}/test"
            }
            ground_truth_view(ground_truth_options)

@dec_func_start_end
def multi_synthesis_image_annotation():

    coco = MultiSynthesis()
    names = ["data"]

    origin_path = "F://custom/_others_/crop"
    save_path = "F://custom/_others_/multi_synthesis"
    background_path = "E://daq/_background_/null/none"

    # origin_path = "F://custom/_seperation_/reside_remove_3"
    # save_path = "F://custom/_seperation_/multi_synthesis_20211130_1"
    # background_path = "E://daq/_background_/null/none"

    # origin_path = "F://ct/_train_/rotate"
    # save_path = "F://ct/_train_/multi_synthesis"
    # background_path = "E://daq/_background_/null/background_fix"

    # origin_path = "E://daq/_train_/crop"
    # save_path = "E://daq/_train_/multi_synthesis_none"
    # background_path = "E://daq/_background_/null/none"

    # origin_path = "E://aixac/_train_/daqdata"
    # save_path = "E://aixac/_train_/daqdata_multi_synthesis_3"
    # background_path = "E://aixac/_background_/null/background_fix"

    make_dir(save_path,{"is_remove":True})

    dir_list_options = {
        "dir_path": origin_path
    }

    fulls, _, _ = load_dir_list(dir_list_options)

    progress = Progress(max_num=len(fulls),work_name=__name__)

    for full in fulls:
        progress.set_work_name(f" = {full}\n")
        progress.update()

        for name in names:
            path_options = {
                "image_path": f"{full}/image",
                "json_path": f"{full}/json/{name}.json",
            }
            coco.change_image_path(path_options)

            objects_name = full.split("/")[-1]
            max_category_count = 10
            if objects_name.find("bag") > -1:
                max_category_count = 5
                
            options = {
                "file_name_header": f"{name}",
                "background_path": background_path,

                "json_path": f"{full}/json/{name}.json",

                "save_image_path": f"{save_path}/{objects_name}/image",
                "save_json_path": f"{save_path}/{objects_name}/json",

                "max_image_count": 400,
                "max_category_count": max_category_count,
                "background_fix_width": 1200,
                "background_fix_height": 1156,
                "inner_point_options": {
                    "padding_width": 10,
                    "padding_height": 10,
                },
                "use_overlap":True,
                "use_sequential":True
            }
            coco.run(options)
            
            ground_truth_options = {
                "image_path": f"{save_path}/{objects_name}/image",
                "json_path": f"{save_path}/{objects_name}/json/{name}.json",
                "gt_path": f"{save_path}/{objects_name}/test"
            }
            ground_truth_view(ground_truth_options)

class SingleSynthesis(Bone):

    def run(self, options):
        '''
        options = {
            "image_path":{image_path},
            "json_path":{json_path},
            "save_image_path":{save_image_path},
            "save_json_path":{save_json_path},
            "ends_with":{ends_with},
            "file_path":{file_path}
        }
        '''
        try:
            image_path = options["image_path"]
            json_path = options["json_path"]

            save_image_path = options["save_image_path"]
            save_json_path = options["save_json_path"]

            ends_with = options["ends_with"]
            file_path = options["file_path"]

            inner_point_options = options["inner_point_options"]

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

            background_fix_point_options = {
                "max_width": 0,
                "max_height": 0,
                "point_x": 0,
                "point_y": 0,
            }

            synthesis_options = {
                "alpha": 1,
                "beta": 1,
                "correction_pixel": 0,
                "max_pixel": 65535
            }

            file_list_options = {
                "ends_with":ends_with,
                "file_path":file_path
            }

            _, background_paths, background_files = load_file_list(file_list_options)
            
            backgrounds = []
            for path, file in zip(background_paths, background_files):
                data = self.load_image(path, file)
                backgrounds.append(data)
            
            for background in backgrounds: 
                max_height, max_width = np.shape(background)

                for image in images:
                    image_id = image["id"]

                    for annotation in annotations:
                        bbox = annotation["bbox"]
                        segmentation = annotation["segmentation"]
                        annotation_image_id = annotation["image_id"]
                        
                        if image_id == annotation_image_id:
                            image_path = image["path"]
                            path, name = self.split_path_name(image_path)
                            
                            data = self.load_image(path, name)
                            point_x, point_y = self.inner_point(
                                background, data, inner_point_options)

                            background_fix_height, background_fix_width = np.shape(background)

                            background_fix_point_options["max_width"] = background_fix_width
                            background_fix_point_options["max_height"] = background_fix_height
                            background_fix_point_options["point_x"] = point_x
                            background_fix_point_options["point_y"] = point_y
                            
                            background_fix_data = background_fix_position(data, background_fix_point_options)
                            
                            synthesis_data = synthesis(background, background_fix_data, synthesis_options)
                            
                            copy_annotation = copy.copy(annotation)
                            copy_annotation["bbox"] = self._bbox(bbox, point_x, point_y)
                            copy_annotation["segmentation"] = self._segmentation(segmentation, point_x, point_y)
                            copy_annotation["id"] = annotation_idx
                            copy_annotation["image_id"] = image_idx
                            
                            json_result["annotations"].append(copy_annotation)
                            
                            annotation_idx += 1

                    name = f"{image_idx}.png"

                    copy_image = copy.copy(image)
                    copy_image["width"] = max_width
                    copy_image["height"] = max_height
                    copy_image["path"] = f"{save_image_path}/{name}"
                    copy_image["file_name"] = name
                    copy_image["id"] = image_idx
                    json_result["images"].append(copy_image)
                    self.save_image(synthesis_data,save_image_path,name)
                    image_idx += 1


            save_options = {
                "load_path":save_json_path,
                "file_name":json_file
            }
            self.save_json(json_result, save_options)
        except Exception as ex:
            logger_exception(ex)
            
    def _bbox(self, bbox, start_x, start_y):
        return [bbox[0]+start_x,bbox[1]+start_y,bbox[2],bbox[3]]

    def _segmentation(self, segmentation, start_x, start_y):
        result = []
        for seg in segmentation:
            seg_array = np.array(seg)
            seg_array[::2] += start_x
            seg_array[1::2] += start_y
            seg_array = np.round_(seg_array, 1)  # 소수점 1자리수에서 반올림
            result.append(seg_array.tolist())
        return result

class MultiSynthesis(Bone):

    def run(self, options):
        '''
        options = {
            "image_path":{image_path},
            "json_path":{json_path},
            "save_image_path":{save_image_path},
            "save_json_path":{save_json_path},
            "max_image_count":{max_image_count},
            "max_category_count":{max_category_count},
            "background_fix_width":{background_fix_width},
            "background_fix_height":{background_fix_height}
            inner_point_options":{
                "padding_width":{padding_width},
                "padding_height":{padding_height}
            }
        }
        '''
        try:
            background_path = options["background_path"]
            json_path = options["json_path"]

            save_image_path = options["save_image_path"]
            save_json_path = options["save_json_path"]

            max_image_count = options["max_image_count"]
            max_category_count = options["max_category_count"]

            background_fix_width = options["background_fix_width"]
            background_fix_height = options["background_fix_height"]

            inner_point_options = options["inner_point_options"]
            use_overlap = options["use_overlap"]
            use_sequential = options["use_sequential"]
            
            make_dir(save_image_path, {"is_remove": True})
            make_dir(save_json_path, {"is_remove": True})

            background_file_list_options = {
                "ends_with": ".png",
                "file_path": background_path
            }
            _, _, background_files = load_file_list(background_file_list_options)
            
            annotation_id_count = 0
            image_id_count = 0

            json_path, json_file = self.split_path_name(json_path)

            load_options = {
                "load_path": json_path,
                "file_name": json_file
            }
            
            json_data = self.load_json(load_options)

            annotations = self.get_annotations(json_data)
            categories = self.get_categories(json_data)
            images = self.get_images(json_data)

            json_result = {
                "images": [],
                "categories": categories,
                "annotations": [],
            }

            for _, background_file in enumerate(background_files):
                
                background = self.load_image(background_path, background_file)

                synthesis_options = {
                    "alpha": 1,
                    "beta": 1,
                    "correction_pixel": 0,
                    "max_pixel": 65535
                }

                background_height, background_width = np.shape(background)

                background_fix_point_options = {
                    "max_width": background_width,
                    "max_height": background_height,
                    "point_x": 0,
                    "point_y": 0,
                }

                image_count = 0
                sequential_count = 0
                
                while image_count < max_image_count:
                # for _ in range(max_image_count):
                    data_list = []
                    annotation_list = []
                    image_id_list = []
                    start_x_list = []
                    start_y_list = []
                    bbox_width_list = []
                    bbox_height_list = []
                    
                    now_count = 0
                    
                    for _ in range(max_category_count):
                        if use_sequential:
                            ann_index = self.sequential_annotations_selected(sequential_count, json_data)
                            sequential_count = copy.copy(ann_index)
                        else:
                            ann_index = self.random_annotations_selected(json_data)
                            
                        annotation = annotations[ann_index]
                        image_id = annotation["image_id"]

                        path, name = self.split_path_name(images[image_id]["path"])
                        data = self.load_image(path, name)
                        
                        start_x, start_y = self.inner_point(background, data, inner_point_options)
                        background_fix_point_options["point_x"] = start_x
                        background_fix_point_options["point_y"] = start_y
                        
                        _,_,width,height = annotation["bbox"]
                        
                        data = background_fix_position(data, background_fix_point_options)
                        
                        is_overlap = False
                        data_bbox_list = {
                            "x_list": start_x_list,
                            "y_list": start_y_list,
                            "width_list": bbox_width_list,
                            "height_list": bbox_height_list,
                        }

                        now_bbox = {
                            "x": int(start_x),
                            "y": int(start_y),
                            "width": int(start_x + width),
                            "height": int(start_y + height)
                        }
                        
                        is_overlap = False
                        if not use_overlap:
                            is_overlap = self.overlap_check(background, data_bbox_list, now_bbox)
                        
                        if not is_overlap:
                            data_list.append(data)
                            annotation_list.append(annotation)
                            image_id_list.append(image_id)
                            
                            start_x_list.append(start_x)
                            start_y_list.append(start_y)
                            
                            bbox_width_list.append(start_x+width)
                            bbox_height_list.append(start_y+height)
                            now_count += 1

                    if max_category_count == now_count:
                        ##
                        null_background = np.full((background_height, background_width), 65535, dtype=np.uint16)
                        for data in data_list:
                            null_background = synthesis(data, null_background, synthesis_options)
                        synthesis_data = synthesis(background, null_background, synthesis_options)

                        background_fix_options = {
                            "max_width": background_fix_width,
                            "max_height": background_fix_height
                        }

                        background_fix_synthesis_data = background_fix_center(synthesis_data, background_fix_options)
                        path, name = self.split_path_name(f"{save_image_path}/{image_id_count}.png")
                        self.save_image(background_fix_synthesis_data, path, name)

                        ##
                        result_images = self.init_image_dataset()
                        result_images["path"] = f"{path}/{name}"
                        result_images["file_name"] = name
                        result_images["width"] = background_fix_width
                        result_images["height"] = background_fix_height
                        result_images["id"] = image_id_count
                        
                        json_result["images"].append(result_images)

                        for image_id, annotation, start_x, start_y in zip(image_id_list, annotation_list, start_x_list, start_y_list):
                            result_annotation = self.init_annotation_dataset()
                            _, _, bbox_width, bbox_height = annotation["bbox"]

                            result_annotation["id"] = annotation_id_count
                            result_annotation["image_id"] = image_id_count
                            result_annotation["category_id"] = annotation["category_id"]

                            seg_size = len(annotation["segmentation"][0])

                            result_annotation["area"] = annotation["area"]
                            result_annotation["bbox"] = self._bbox(background_fix_width, background_fix_height, background_width, background_height, start_x, start_y, bbox_width, bbox_height)
                            result_annotation["segmentation"] = self._segmentation(seg_size, annotation, background_fix_width, background_fix_height, background_width, background_height, start_x, start_y)

                            json_result["annotations"].append(result_annotation)

                            annotation_id_count += 1
                        image_id_count += 1

                        image_count += 1

            save_options = {
                "load_path":save_json_path,
                "file_name":json_file
            }
            self.save_json(json_result, save_options)
        except Exception as ex:
            logger_exception(ex)
            
    def _bbox(self, background_fix_width, background_fix_height, background_width, background_height, start_x, start_y, bbox_width, bbox_height):
        cal_background_fix_start_x = background_fix_width//2 - background_width//2 + start_x
        cal_background_fix_start_y = background_fix_height//2 - background_height//2 + start_y

        return [cal_background_fix_start_x, cal_background_fix_start_y, bbox_width, bbox_height]

    def _segmentation(self, seg_size, annotation, background_fix_width, background_fix_height, background_width, background_height, start_x, start_y):
        cal_background_fix_start_x = background_fix_width//2 - background_width//2 + start_x
        cal_background_fix_start_y = background_fix_height//2 - background_height//2 + start_y

        seg_size = len(annotation["segmentation"][0])
        result = []

        for s in range(0, seg_size, 2):
            cal_annotation_x = annotation["segmentation"][0][s] + cal_background_fix_start_x
            cal_annotation_y = annotation["segmentation"][0][s +1] + cal_background_fix_start_y

            result.append(cal_annotation_x)
            result.append(cal_annotation_y)
        return [result]
    
    def random_annotations_selected(self, json_data):
        annotations = self.get_annotations(json_data)
        size = len(annotations)
        result = random.randrange(0, size)
        return result

    def sequential_annotations_selected(self, sequential_count, json_data):
        annotations = self.get_annotations(json_data)
        size = len(annotations)
        sequential_count += 1
        if sequential_count >= size:
            sequential_count = 0 
        result = sequential_count
        return result

    def overlap_check(self, background, data_list, now_bbox):
        result = False
        
        background_height, background_width = np.shape(background)

        x_list = data_list["x_list"]
        y_list = data_list["y_list"]
        width_list = data_list["width_list"]
        height_list = data_list["height_list"]
        
        background_now = np.zeros( (background_height, background_width) ,dtype=np.uint8 )

        now_x = int(now_bbox["x"]) 
        now_y = int(now_bbox["y"])
        now_width = int(now_bbox["width"])
        now_height = int(now_bbox["height"])
        
        background_now[now_y:now_height,now_x:now_width] = 1
        
        for x,y,width,height in zip(x_list,y_list,width_list,height_list):
            background_before = np.zeros( (background_height, background_width) )
            background_before[int(y):int(height),int(x):int(width)] = 1
            compare = background_before + background_now
            is_overlap = np.max(compare)
            if is_overlap > 1:
                result = True
                break

        return result

