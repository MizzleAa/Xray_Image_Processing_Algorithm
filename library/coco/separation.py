from library.utils.header import *
from library.utils.decorator import *
from library.utils.io import *
from library.utils.log import *
from library.utils.progress import *

from library.coco.groundtruth import *
from library.coco.define import *

@dec_func_start_end
def seperation_image_annotation():
    """Background 클레스 사용 예제

    .. note::

        Background 클레스 사용 예시 입니다. 해당 클레스를 사용하기 위한 파일 전처리가 필요 하며, 해당 파일이 올바르게 수행되었는지 검증(ground truth)이 필요합니다.
        
    """
    seperation = Seperation()
    names = ["data"]

    # origin_path = "E://aixac/_train_/origin"
    # save_path = "E://aixac/_train_/seperation"
    
    # origin_path = "F://custom/_train_/origin"
    # save_path = "F://custom/_train_/seperation"
    
    origin_path = "F://custom/_source_/ct_to_16bit"
    save_path = "F://custom/_source_/ct_seperation"

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
            seperation.change_image_path(path_options)
            
            objects_name = full.split("/")[-1]
            options = {
                "image_path": f"{full}/image",
                "json_path": f"{full}/json/{name}.json",
                "save_path": f"{save_path}/{objects_name}",
            }
            seperation.run(options)
        # break
            

class Seperation(Bone):
    """배경화면을 두어 정재된 데이터를 화면의 정중앙으로 위치를 이동시킵니다.
    
    .. note::

        일정 크기의 배경 맵을 지정하여 해당 맵의 정중앙으로 객체를 이동시킵니다.

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

            save_path = options["save_path"]

            make_dir(save_path)

            json_path, json_file = self.split_path_name(json_path)
            
            json_data_options = {
                "load_path":json_path,
                "file_name":json_file
            }
            json_data = self.load_json(json_data_options)

            images = self.get_images(json_data)
            annotations = self.get_annotations(json_data)
            categories = self.get_categories(json_data)
            
            for category in categories:
                
                category_save_path = f"{save_path}/{category['name']}"
                make_dir(category_save_path)
                make_dir(f"{category_save_path}/image")
                make_dir(f"{category_save_path}/json")
                
                json_result = {
                    "images":[],
                    "categories":[copy.copy(category)],
                    "annotations":[],
                }
                
                annotation_id = 0
                image_id = 0
                
                for annotation in annotations:
                    if category["id"] == annotation["category_id"]:
                        for image in images:
                            if annotation["image_id"] == image["id"] and category["id"] == annotation["category_id"]:
                                # print(annotation["image_id"], image["id"], category["id"], annotation["category_id"])
                                copy_annotation = copy.copy(annotation)
                                copy_image = copy.copy(image)
                                
                                origin_path = f"{image_path}/{image['file_name']}"
                                copy_path = f"{save_path}/{category['name']}/image/{image_id}_{annotation_id}.png"
                                
                                shutil.copy(origin_path,copy_path)
                                
                                copy_image["id"] = image_id
                                copy_image["path"] = copy_path
                                copy_image["file_name"] = f"{image_id}_{annotation_id}.png"
                                
                                copy_annotation["id"] = annotation_id
                                copy_annotation["image_id"] = image_id
                                
                                json_result["images"].append(copy_image)
                                json_result["annotations"].append(copy_annotation)
                                
                                annotation_id+=1
                                image_id+=1
                                
                                break

                save_options = {
                    "load_path":f"{save_path}/{category['name']}/json",
                    "file_name":json_file
                }
                # print(save_options)
                self.save_json(json_result,save_options)
                
                ground_truth_options = {
                    "image_path": f"{save_path}/{category['name']}/image",
                    "json_path": f"{save_path}/{category['name']}/json/data.json",
                    "gt_path": f"{save_path}/{category['name']}/test"
                }
                ground_truth_view(ground_truth_options)
                # break
            
        except Exception as ex:
            logger_exception(ex)

    def _bbox(self, bbox, width, height):
        """annotation의 bbox를 반환합니다.

        Args:
            bbox (list) : x,y,width,height에 대한 정보가 들어있는 리스트입니다.
            width (int) : image의 가로 길이 입니다.
            height (int) : image의 세로 길이 입니다.

        Return: 
            result (list) : [x,y,width,height] 로 반환합니다.

        Raise:
            None
        
        """
        return [bbox[0]+width,bbox[1]+height,bbox[2],bbox[3]]

    def _segmentation(self, segmentation, width, height):
        """annotation을 반환합니다.

        Args:
            segmentation (list) : [x1,y1,x2,y2, ...] 형식의 리스트 입니다.
            width (int) : image의 가로 길이 입니다.
            height (int) : image의 세로 길이 입니다.

        Return: 
            result (list) : [x1,y1,x2,y2, ...] 로 반환합니다.

        Raise:
            None
        
        """
        result = []
        for seg in segmentation:
            seg_array = np.array(seg)
            seg_array[::2] += width
            seg_array[1::2] += height
            seg_array = np.round_(seg_array, 1)
            result.append(seg_array.tolist())
        return result
