from library.utils.header import *
from library.utils.decorator import *
from library.utils.log import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

from library.images.crop import *
from library.images.mask import *

@dec_func_start_end
def crop_image_annotation():
    """crop 클레스 사용 예제

    .. note::

        crop 클레스 사용 예시 입니다. 해당 클레스를 사용하기 위한 파일 전처리가 필요 하며, 해당 파일이 올바르게 수행되었는지 검증(ground truth)이 필요합니다.
        
    """
    coco = Crop()
    names = ["data"]

    origin_path = "./sample/xray/example_10/object/origin"
    save_path = "./sample/xray/example_10/object/crop"

    # origin_path = "F://custom/_others_/null/origin"
    # save_path = "F://custom/_others_/null/crop"
    
    # origin_path = "F://ct/_train_/rate"
    # save_path = "F://ct/_train_/crop"
    
    # origin_path = "F://custom/_others_/background_fix"
    # save_path = "F://custom/_others_/crop"

    # origin_path = "F://custom/_seperation_/20211129_gray_to_resider"
    # save_path = "F://custom/_seperation_/20211129_resider_to_crop"
    
    # origin_path = "F://custom/_seperation_/__reside"
    # save_path = "F://custom/_seperation_/__crop"

    # origin_path = "F://daq/_train_/origin"
    # save_path = "F://daq/_train_/crop"

    # origin_path = "E://aixac/_train_/seperation"
    # save_path = "E://aixac/_train_/crop"

    # origin_path = "E://ct/_train_/seperation"
    # save_path = "E://ct/_train_/crop"

    # origin_path = "E://daq/_train_/origin"
    # save_path = "E://daq/_train_/crop"

    make_dir(save_path, options={"is_remove":True})

    dir_list_options = {
        "dir_path": origin_path
    }

    paths, _, _ = load_dir_list(dir_list_options)
    
    progress = Progress(max_num=len(paths),work_name=__name__)
    
    for path in paths:
        #progress.set_work_name(f" = {path}\n")
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
            }
            coco.run(options)

            ground_truth_options = {
                "image_path": f"{save_path}/{objects_name}/image",
                "json_path": f"{save_path}/{objects_name}/json/{name}.json",
                "gt_path": f"{save_path}/{objects_name}/test"
            }
            ground_truth_view(ground_truth_options)

class Crop(Bone):
    """정제된 영역을 기준으로 image를 잘라 잘린 image를 생성합니다.
    
    .. note::

        정제된 데이터의 영역을 잘라 해당 크기만큼의 잘린 image를 생성합니다.

    """

    def run(self, options):
        """해당 함수는 다음과 같이 정의됩니다.
        병합된 image,json 데이터는 하나의 폴더안에 저장됩니다. 

        Args:
            options (dict) : 입력값을 정의합니다.
        
        Return: 
            None

        Raise:
            fileio - 파일 입출력에 관한 에러

        options의 구성은 다음과 같이 정의 됩니다.

        .. code-block:: JSON

            "options": {
                "file_path_list": "image 파일 경로 목록",
                "json_list": "json 파일 경로 목록",
                "save_file_path": "image 파일 저장 경로",
                "save_json_path": "json 파일 저장 경로"
                "max_width": "고정시킬 image의 가로 크기",
                "max_height": "고정시킬 image의 세로 크기",
            }

        예시

        .. code-block:: python
        
            background = Background()
            options = {
                "file_path_list": file_path_list,
                "json_list": json_list,
                "save_file_path": f"{save_path}/image",
                "save_json_path": f"{save_path}/json",
                "max_width": 700,
                "max_height": 570,
            }
            background.run(options)

        """

        try:
            image_path = options["image_path"]
            json_path = options["json_path"]

            save_image_path = options["save_image_path"]
            save_json_path = options["save_json_path"]

            make_dir(save_image_path,options={"is_remove":True})
            make_dir(save_json_path,options={"is_remove":True})

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

            crop_options ={
                "x":0,
                "y":0,
                "width":0,
                "height":0
            }

            for annotation in annotations:
                bbox = annotation["bbox"]
                segmentation = annotation["segmentation"]
                annotation_image_id = annotation["image_id"]
                
                for img_idx, image in enumerate(images):
                    image_id = image["id"]
                    if image_id == annotation_image_id:
                        image_path = image["path"]
                        path, name = self.split_path_name(image_path)
                        
                        data = self.load_image(path, name)
                        
                        # data = self.load_image(path, name) * 255
                        # data = data.astype(np.uint8)
                        # print(np.min(data), np.max(data))

                        crop_options["x"] = bbox[0]
                        crop_options["y"] = bbox[1]
                        crop_options["width"] = bbox[2]
                        crop_options["height"] = bbox[3]
                        #
                        crop_data = crop_rectangle(data, crop_options)
                        
                        #
                        name = f"{image_idx}.jpg"

                        copy_annotation = copy.copy(annotation)
                        copy_image = copy.copy(image)

                        copy_image["id"] = image_idx
                        copy_image["height"], copy_image["width"] = np.shape(crop_data)[0:2]
                        copy_image["path"] = f"{save_image_path}/{name}"
                        copy_image["file_name"] = name

                        bbox_width = bbox[2]
                        bbox_height = bbox[3]
                        segmentation_x = bbox[0]
                        segmentation_y = bbox[1]
                        ##
                        copy_annotation["bbox"] = self._bbox(bbox_width, bbox_height)
                        copy_annotation["segmentation"] = self._segmentation(segmentation,segmentation_x,segmentation_y)
                        ##
                        copy_annotation["id"] = annotation_idx
                        copy_annotation["image_id"] = image_idx

                        json_result["images"].append(copy_image)
                        json_result["annotations"].append(copy_annotation)
                        
                        crop_data = mask_polygon(crop_data, copy_annotation["segmentation"])

                        save_options={"dtype": "uint8", "end_pixel": 255}
                        self.save_image(crop_data,save_image_path,name,save_options)

                        annotation_idx += 1
                        image_idx += 1

                        #del images[img_idx]
                        #break

            save_options = {
                "load_path":save_json_path,
                "file_name":json_file
            }

            self.save_json(json_result, save_options)
        except Exception as ex:
            logger_exception(ex)
            
    def _bbox(self, width, height):
        return [0,0,width,height]

    def _segmentation(self, segmentation, width, height):
        result = []
        for seg in segmentation:
            seg_array = np.array(seg)
            seg_array[::2] -= width
            seg_array[1::2] -= height
            seg_array = np.round_(seg_array, 1)  # 소수점 1자리수에서 반올림
            result.append(seg_array.tolist())
        return result